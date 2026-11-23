# Lore samples (fine-tuned 100M model, unseen tags)

Taken unedited from notebook 06, section 5. ✅ = passed every automatic check. Note the logic slips discussed in [docs/07](../docs/07_results_and_lessons.md).

```text
metric                   base  fine-tuned  teacher
passes_all_checks       0.133       0.933    1.000
uses_name               0.533       0.978    1.000
distinct_4grams         0.724       0.970    0.942
top_opening_share       0.111       0.067    0.021
median_words          173.000     119.000  117.000

▶ type: planet | name: Oris | climate: tidally locked | size: giant | life: none | danger: extreme | feature: a strange repeating signal | mood: melancholic | located_in: Morenae | length: short
Oris, a giant tidally locked planet in the Morenae system, is a world of stark contrasts. Its dayside is perpetually bathed in sunlight, while the nightside hosts a thick, toxic atmosphere that glows with eerie green hues. The night side, on the other hand, remains shrouded in perpetual twilight. What makes Oris truly memorable are the strange repeating signal emanating from its polar regions—a haunting, rhythmic pulse that seems to shift and change with each passing hour. This unexplained phenomenon adds a layer of mystery and unease to the otherwise melancholic planet.
✅

▶ type: star_system | name: Ixisia Expanse | star: white dwarf | planet_count: 2 | status: contested | mood: eerie | length: medium
In the Ixisia Expanse, a faint light casts a pale glow over two worlds:

The first world, Aethoria, is a desolate wasteland where life persists in impossible conditions. Its surface is scarred by ancient impact craters and now covered in shifting dunes that shift under the dim, blue sun. 

Aethoria, a gas giant with a tumultuous atmosphere, has a ring system composed of dark minerals that reflect light into surreal patterns. The gas giant's intense radiation has created storms that unleash torrents of violent weather. Despite its hostile environment, the Ixisia Expanse is a place where secrets are whispered and truths are buried beneath the surface.

The fourth planet, Kaelon, is a barren rock with a thick, sulfurous atmosphere that glows with an eerie blue light. Its surface is dotted with ancient ruins scattered across its landscape, their purpose unknown to outsiders. 

Despite its hostile conditions, Ixisia Expanse remains a place of both intrigue and mystery, where the echoes of long-forgotten civilizations echo through the void.
✅

▶ type: faction | name: the Children of Eskaear | kind: nomad fleet | goal: freedom | methods: faith | alignment: ruthless | mood: ominous | homeworld: Eskra Minor | length: medium
The Children of Eskaear, a nomad fleet known as the Eskra Minor, seek freedom from their homeworld’s oppressive constraints. They operate in the shadowy depths of uncharted space, seeking to reclaim ancient knowledge and independence. Their ships, sleek and deadly, flicker with an ominous glow, always ready to defend their ideals.

Their goal is simple: To achieve freedom. The Children do not have a fleet—each member a shard of their own, capable of altering the course of events through force or coercion. They maintain a shroud of secrecy, whispering warnings and promises that others would see them as rebels against their oppressive rule.

Eskra Minor’s nightmarish temples hum with the energy of countless lives saved and restarted. Each ship is a testament to their unyielding resolve. The Children of Eskaear are not just nomads; they are masters of the cosmos, capable of shaping the fabric of reality itself.
✅

▶ type: species | name: the Hessi | body: insectoid | habitat: desert | society: democratic | temperament: isolationist | unique_trait: speaks only in music | mood: unsettling | length: medium
The Hessi, an insectoid species native to the arid desert planet of Valtor, have elongated, segmented bodies that shimmer with iridescent hues. Their segmented bodies are covered in iridescent scales, allowing them to absorb sunlight through specialized chlorophyll-like structures that sustain their vibrant colors. These fibers allow them to blend seamlessly into their surroundings, while their eyes—large, round lenses connected by thin, translucent wings—remind them of the harshness of their environment.

In a society structured around democracy and consensus, Hessi communities are built on shared knowledge and mutual respect. They maintain an isolationist stance, rarely venturing beyond their desert colonies, which often leaves other species feeling unsettled and isolated. To other species, the Hessi appear as enigmatic sentinels, their presence both beautiful and unsettling. Their music is a haunting melody, hauntingly beautiful yet utterly foreign to the most distant minds.
✅

▶ type: ship_log | name: Belison's Wake | author: Idris Ulon | situation: rumours of mutiny | day: 216 | mood: calm | length: short
Day 216. Belison's Wake. The air hums with tension, a quiet confidence that lingers in the corridors. Rumors of mutiny swirl like dark clouds on the horizon, but tonight, I keep my eyes open, knowing we hold the weight of countless whispers. The ship feels like a place where whispers turn to violence, and I can almost taste the fear and dread in my soul.
✅
```
