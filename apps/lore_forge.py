#!/usr/bin/env python3
"""lore_forge.py — generate space lore with your own 100M model (GTX 1650, any GPU, or CPU).

Run from the unzipped export folder:
    python lore_forge.py            # web app at http://127.0.0.1:7860
    python lore_forge.py --cli      # print a few random lore pieces in the terminal
Options: --model_dir DIR  --cpu  --port 7860  --share
"""
import argparse
import os
import random
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- pre-flight
def preflight(args):
    d = os.path.abspath(args.model_dir)
    problems = [f"missing {f} in {d}" for f in ("model.pt", "tokenizer.json", "model_v2.py", "lore_schema.py", "world_schema.json")
                if not os.path.exists(os.path.join(d, f))]
    for mod, hint in (("torch", "pip install torch --index-url https://download.pytorch.org/whl/cu126"),
                      ("tokenizers", "pip install tokenizers")):
        try:
            __import__(mod)
        except ImportError:
            problems.append(f"{mod} not installed: {hint}")
    if not args.cli:
        try:
            import gradio  # noqa: F401
        except ImportError:
            problems.append("gradio not installed: pip install gradio   (or run with --cli)")
    if problems:
        print("Pre-flight failed:\n- " + "\n- ".join(problems)); sys.exit(1)
    import torch
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    if device == "cuda":
        p = torch.cuda.get_device_properties(0); print(f"GPU: {p.name} ({p.total_memory/1e9:.1f} GB)")
    elif not args.cpu:
        print("No CUDA GPU visible — using the CPU (fine for a 100M model).")
    return d, device


# ---------------------------------------------------------------- engine
class LoreEngine:
    def __init__(self, model_dir, device):
        import torch
        from tokenizers import Tokenizer
        sys.path.insert(0, model_dir)
        import lore_schema
        from model_v2 import load_exported
        self.torch, self.device, self.ls = torch, device, lore_schema
        self.model, meta = load_exported(os.path.join(model_dir, "model.pt"), device,
                                         torch.float16 if device == "cuda" else torch.float32)
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        sp = meta["special_tokens"]
        self.USER, self.ASSIST, self.END, self.EOT = sp["<|user|>"], sp["<|assistant|>"], sp["<|end|>"], sp["<|endoftext|>"]
        self.NL = self.tok.encode("\n").ids
        self.S = lore_schema.load_schema(os.path.join(model_dir, "world_schema.json"))
        self.rng = random.Random()
        self.names = lore_schema.NameGen(self.S, self.rng)
        self.lock = threading.Lock()
        print(f"Loaded {self.model.num_params()/1e6:.0f}M-parameter lore model on {device}")

    # --- entities
    def random_entity(self, etype):
        return self.ls.sample_entity(self.S, self.rng, self.names, etype)

    def make_entity(self, etype, name=None, links=None, length=None):
        """Random entity of a type, with chosen name/links, in the exact tag order used in training."""
        base = self.random_entity(etype)
        T = self.S["types"][etype]
        e = {"type": etype, "name": name or base["name"]}
        for f in T["fields"]:
            if f in base:
                e[f] = base[f]
        for rel in T["links"]:
            if links and rel in links:
                e[rel] = links[rel]
        e["length"] = length or base["length"]
        return e

    # --- generation
    def stream(self, tags, max_new=260, temperature=0.8, top_k=50, top_p=0.9, repetition_penalty=1.1):
        torch = self.torch
        ids = [self.USER] + self.NL + self.tok.encode(tags).ids + [self.END] + self.NL + [self.ASSIST] + self.NL
        idx = torch.tensor([ids], device=self.device)
        out = []
        with self.lock:
            for t in self.model.generate_stream(idx, int(max_new), temperature=float(temperature), top_k=int(top_k),
                                                top_p=float(top_p), repetition_penalty=float(repetition_penalty),
                                                stop_ids=(self.END, self.EOT)):
                out.append(t)
                yield self.tok.decode(out).strip()

    def generate(self, tags, **kw):
        text = ""
        for text in self.stream(tags, **kw):
            pass
        return text

    def check(self, tags, text):
        try:
            return self.ls.check_lore(self.S, self.ls.parse_tags(tags), text)
        except Exception as e:  # malformed hand-edited tags
            return [f"could not check tags ({e})"]


def field_help(S, etype):
    T = S["types"][etype]
    lines = [f"**{etype}** — {T['description']}", ""]
    for f, spec in T["fields"].items():
        vals = ", ".join(spec) if isinstance(spec, list) else ("a person's name" if spec == "@character" else "a number")
        lines.append(f"- `{f}`{' (optional)' if f in T['optional'] else ''}: {vals}")
    if T["links"]:
        lines.append("- links (optional): " + ", ".join(f"`{r}` → a {t} name" for r, t in T["links"].items()))
    lines.append("- `length`: short or medium")
    return "\n".join(lines)


# ---------------------------------------------------------------- star system
def star_system_items(engine, n_planets_max=4):
    """A consistent set of tag lines: one system, its planets, factions, a species and a station."""
    system = engine.make_entity("star_system", length="medium")
    n = min(int(system.get("planet_count", 3)), n_planets_max)
    planets = [engine.make_entity("planet", links={"located_in": system["name"]}) for _ in range(n)]
    homes = [p["name"] for p in planets] or [engine.names.new("planet")]
    items = [("Star system", system)] + [(f"Planet {i+1}", p) for i, p in enumerate(planets)]
    items += [(f"Faction {i+1}", engine.make_entity("faction", links={"homeworld": engine.rng.choice(homes)})) for i in range(2)]
    items.append(("Native species", engine.make_entity("species", links={"homeworld": engine.rng.choice(homes)})))
    items.append(("Station", engine.make_entity("station", links={"located_in": system["name"]})))
    return system, items


# ---------------------------------------------------------------- web UI
def build_ui(engine):
    import gradio as gr
    types = list(engine.S["types"])
    os.makedirs(os.path.join(HERE, "codex"), exist_ok=True)

    def randomize(etype):
        return engine.ls.to_tags(engine.random_entity(etype)), field_help(engine.S, etype)

    def forge(tags, temperature, top_p, rep, max_new):
        if not tags.strip():
            yield "Pick a type and press 🎲 Randomize (or type tags).", ""; return
        text = ""
        for text in engine.stream(tags.strip(), max_new, temperature, 50, top_p, rep):
            yield text, ""
        probs = engine.check(tags.strip(), text)
        yield text, ("✅ follows the tags" if not probs else "⚠ " + "; ".join(probs))

    def forge_system(temperature, top_p, rep):
        system, items = star_system_items(engine)
        md = f"# The {system['name']}\n"
        yield md, None
        for title, e in items:
            tags = engine.ls.to_tags(e)
            md += f"\n## {title}: {e['name']}\n<sub>`{tags}`</sub>\n\n"
            text = ""
            for text in engine.stream(tags, 260, temperature, 50, top_p, rep):
                yield md + text, None
            md += text + "\n"
        path = os.path.join(HERE, "codex", f"{system['name'].replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        yield md, path

    with gr.Blocks(title="Lore Forge") as demo:
        gr.Markdown("# 🌌 Lore Forge\nSpace lore written by your own 100M-parameter model, trained from scratch.")
        with gr.Accordion("Generation settings", open=False):
            temperature = gr.Slider(0.1, 1.5, 0.8, step=0.05, label="Temperature (higher = wilder)")
            top_p = gr.Slider(0.5, 1.0, 0.9, step=0.05, label="Top-p")
            rep = gr.Slider(1.0, 1.5, 1.1, step=0.01, label="Repetition penalty")
            max_new = gr.Slider(64, 400, 260, step=8, label="Max length (tokens)")

        with gr.Tab("Forge one"):
            with gr.Row():
                etype = gr.Dropdown(types, value="planet", label="Type")
                roll = gr.Button("🎲 Randomize")
            tags = gr.Textbox(label="Tags (edit freely)", lines=2)
            go = gr.Button("✨ Forge", variant="primary")
            out = gr.Textbox(label="Lore", lines=10)
            status = gr.Markdown()
            help_md = gr.Markdown(field_help(engine.S, "planet"))
            roll.click(randomize, etype, [tags, help_md])
            etype.change(randomize, etype, [tags, help_md])
            go.click(forge, [tags, temperature, top_p, rep, max_new], [out, status])

        with gr.Tab("Generate a star system"):
            gr.Markdown("One click: a star system, its planets, two factions, a native species and a station — "
                        "all linked by name. The result is saved as a Markdown codex in the `codex/` folder.")
            sys_btn = gr.Button("🪐 Generate star system", variant="primary")
            codex = gr.Markdown()
            codex_file = gr.File(label="Codex file")
            sys_btn.click(forge_system, [temperature, top_p, rep], [codex, codex_file])

        demo.load(randomize, etype, [tags, help_md])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", default=HERE)
    ap.add_argument("--cli", action="store_true", help="print random lore in the terminal")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--no_browser", action="store_true")
    args = ap.parse_args()
    model_dir, device = preflight(args)
    engine = LoreEngine(model_dir, device)
    if args.cli:
        for etype in ("planet", "faction", "ship_log"):
            tags = engine.ls.to_tags(engine.random_entity(etype))
            text = engine.generate(tags)
            probs = engine.check(tags, text)
            print(f"\n▶ {tags}\n{text}\n{'✅ follows the tags' if not probs else '⚠ ' + '; '.join(probs)}")
        return
    build_ui(engine).queue().launch(server_name="127.0.0.1", server_port=args.port, share=args.share,
                                    inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
