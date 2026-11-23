#!/usr/bin/env python3
"""chat_app.py — ask your own model questions in the browser (or the terminal).

Works on a GTX 1650 (needs ~0.3 GB of GPU memory), any other NVIDIA GPU, or the CPU.

Usage (run from the unzipped export folder):
    python chat_app.py              # opens http://127.0.0.1:7860
    python chat_app.py --cli        # chat in the terminal instead
Options: --model_dir DIR  --cpu  --port 7860  --share
"""
import argparse
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
QA_TEMPLATE = "Read the text and answer the question.\n\nText: {passage}\n\nQuestion: {question}"   # same as SQuAD fine-tuning


# ---------------------------------------------------------------- pre-flight
def preflight(args):
    problems = []
    model_dir = os.path.abspath(args.model_dir)
    for f in ("model.pt", "tokenizer.json", "model_v2.py"):
        if not os.path.exists(os.path.join(model_dir, f)):
            problems.append(f"missing {f} in {model_dir} (unzip the export and run from that folder, or use --model_dir)")
    try:
        import torch
    except ImportError:
        problems.append("PyTorch not installed: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        torch = None
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        problems.append("tokenizers not installed: pip install tokenizers")
    if not args.cli:
        try:
            import gradio  # noqa: F401
        except ImportError:
            problems.append("gradio not installed: pip install gradio   (or run with --cli)")
    if problems:
        print("Pre-flight failed:\n- " + "\n- ".join(problems))
        sys.exit(1)

    device = "cpu"
    if not args.cpu:
        if torch.cuda.is_available():
            device = "cuda"
            p = torch.cuda.get_device_properties(0)
            print(f"GPU: {p.name} ({p.total_memory / 1e9:.1f} GB)")
        else:
            print("No CUDA GPU visible to PyTorch — using the CPU (still fine for a 100M model).\n"
                  "  For the GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
    return model_dir, device


# ---------------------------------------------------------------- model wrapper
class Engine:
    def __init__(self, model_dir, device):
        import torch
        from tokenizers import Tokenizer
        sys.path.insert(0, model_dir)
        from model_v2 import load_exported

        self.torch, self.device = torch, device
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model, meta = load_exported(os.path.join(model_dir, "model.pt"), device, dtype)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        sp = meta["special_tokens"]
        self.USER, self.ASSIST, self.END, self.EOT = (sp["<|user|>"], sp["<|assistant|>"],
                                                      sp["<|end|>"], sp["<|endoftext|>"])
        self.NL = self.tok.encode("\n").ids
        self.ctx = meta["config"]["block_size"]
        self.lock = threading.Lock()        # one answer at a time
        assert self.tok.get_vocab_size() <= meta["config"]["vocab_size"], "tokenizer and model don't match"
        print(f"Loaded {self.model.num_params() / 1e6:.0f}M-parameter model on {device} | context {self.ctx} tokens")

    def _turn(self, role, text):
        body = self.tok.encode(text).ids
        return ([self.USER] if role == "user" else [self.ASSIST]) + self.NL + body + [self.END] + self.NL

    def build_prompt(self, history, user_msg, max_new):
        """Chat format used in fine-tuning; drops the oldest turns if the context would overflow."""
        budget = self.ctx - max_new - len(self.NL) - 1
        last = self._turn("user", user_msg)
        if len(last) > budget:                                   # very long question: keep its end
            body = self.tok.encode(user_msg).ids[-(budget - 2 * len(self.NL) - 2):]
            last = [self.USER] + self.NL + body + [self.END] + self.NL
        turns = [self._turn(m["role"], m["content"]) for m in history]
        while turns and sum(map(len, turns)) + len(last) > budget:
            turns.pop(0)
        return [t for turn in turns for t in turn] + last + [self.ASSIST] + self.NL

    def stream(self, prompt_ids, max_new=200, temperature=0.7, top_k=40, top_p=0.9, repetition_penalty=1.1):
        torch = self.torch
        idx = torch.tensor([prompt_ids], device=self.device)
        out = []
        with self.lock:
            for tid in self.model.generate_stream(idx, int(max_new), temperature=float(temperature),
                                                  top_k=int(top_k), top_p=float(top_p),
                                                  repetition_penalty=float(repetition_penalty),
                                                  stop_ids=(self.END, self.EOT)):
                out.append(tid)
                yield self.tok.decode(out).strip()
        if not out:
            yield "(no answer — try a higher temperature)"


def as_text(content):
    """Gradio versions differ in how they store message content; always return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text") or content.get("content") or ""
    if isinstance(content, (list, tuple)):
        return " ".join(as_text(c) for c in content)
    return str(content or "")


# ---------------------------------------------------------------- web UI
def build_ui(engine):
    import gradio as gr

    def make_chatbot():
        try:
            return gr.Chatbot(type="messages", height=430, label="Chat")
        except TypeError:                       # newer Gradio: messages format is the only format
            return gr.Chatbot(height=430, label="Chat")

    def user_submit(message, history):
        history = list(history or [])
        if message and message.strip():
            history.append({"role": "user", "content": message.strip()})
        return "", history

    def bot_reply(history, temperature, top_k, top_p, rep, max_new):
        history = [{"role": m["role"], "content": as_text(m["content"])} for m in (history or [])]
        if not history or history[-1]["role"] != "user":
            yield history
            return
        prompt = engine.build_prompt(history[:-1], history[-1]["content"], int(max_new))
        history.append({"role": "assistant", "content": ""})
        for partial in engine.stream(prompt, max_new, temperature, top_k, top_p, rep):
            history[-1]["content"] = partial
            yield history

    def answer_from_text(passage, question, temperature, top_k, top_p, rep, max_new):
        if not passage.strip() or not question.strip():
            yield "Paste a text and type a question."
            return
        prompt = engine.build_prompt([], QA_TEMPLATE.format(passage=passage.strip(), question=question.strip()),
                                     int(max_new))
        yield from engine.stream(prompt, max_new, temperature, top_k, top_p, rep)

    with gr.Blocks(title="My LLM") as demo:
        gr.Markdown(f"## My {engine.model.num_params() / 1e6:.0f}M-parameter language model\n"
                    "Trained from scratch. It is small: expect fluent answers that are sometimes wrong. "
                    "**Answer from a text** works best — the answer is in the text you give it.")
        with gr.Accordion("Generation settings", open=False):
            temperature = gr.Slider(0.0, 1.5, 0.7, step=0.05, label="Temperature (0 = always pick the most likely word)")
            top_k = gr.Slider(1, 200, 40, step=1, label="Top-k (sample only from the k most likely words)")
            top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-p")
            rep = gr.Slider(1.0, 1.5, 1.1, step=0.01, label="Repetition penalty (higher = repeats less)")
            max_new = gr.Slider(16, 512, 200, step=8, label="Max answer length (tokens)")
        settings = [temperature, top_k, top_p, rep, max_new]

        with gr.Tab("Chat"):
            chatbot = make_chatbot()
            msg = gr.Textbox(placeholder="Ask a question and press Enter…", show_label=False)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            gr.Examples(["What is photosynthesis?", "Who was Albert Einstein?",
                         "Write a short story about a dog who finds a hat.", "Give me three tips for studying."],
                        inputs=msg)
            msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_reply, [chatbot, *settings], chatbot)
            send.click(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_reply, [chatbot, *settings], chatbot)
            clear.click(lambda: [], None, chatbot)

        with gr.Tab("Answer from a text"):
            passage = gr.Textbox(lines=8, label="Text",
                                 placeholder="Paste a paragraph, e.g. from Wikipedia or your notes…")
            question = gr.Textbox(label="Question", placeholder="What does the text say about…?")
            ask = gr.Button("Answer", variant="primary")
            answer = gr.Textbox(label="Answer", lines=4)
            ask.click(answer_from_text, [passage, question, *settings], answer)
            question.submit(answer_from_text, [passage, question, *settings], answer)
    return demo


# ---------------------------------------------------------------- terminal mode
def run_cli(engine):
    print("\nType a question (or 'clear' to reset, 'quit' to exit).\n")
    history = []
    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q.lower() in ("quit", "exit"):
            break
        if q.lower() == "clear":
            history = []; print("(history cleared)"); continue
        if not q:
            continue
        prompt = engine.build_prompt(history, q, 200)
        shown, answer = 0, ""
        print("bot > ", end="", flush=True)
        for answer in engine.stream(prompt):
            print(answer[shown:], end="", flush=True); shown = len(answer)
        print("\n")
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default=HERE)
    ap.add_argument("--cli", action="store_true", help="terminal chat instead of the web page")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public Gradio link (anyone with it can use your model)")
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()
    model_dir, device = preflight(args)
    engine = Engine(model_dir, device)
    if args.cli:
        run_cli(engine)
    else:
        build_ui(engine).queue().launch(server_name="127.0.0.1", server_port=args.port,
                                        share=args.share, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
