# Figure text style and text content adjustment

All paper line plots, grid plots, matched-recall plots and bar charts read
their non-legend text settings from:

`5_Plot/figure_text_style.json`

The same file contains both:

1. numeric text-style settings under the top-level keys;
2. human-readable labels under the `content` object.

## 1. Text style

Effective font size:

```
effective = base_size * global_scale * <kind>_scale
```

where `<kind>` is `label`, `tick` or `title`.  Font weights are also read from
the JSON file.  The legend sizes are deliberately not changed (the legends
were already clear in the manuscript layout).

Example:

```json
{
  "global_scale": 1.25,
  "label_scale": 1.0,
  "tick_scale": 1.1,
  "title_scale": 1.0,
  "label_font_weight": "bold",
  "tick_font_weight": "bold",
  "title_font_weight": "bold",
  "content": {
    "bar_memory_ylabel": "Index Memory",
    "bar_build_time_ylabel": "Index Build Time (s)",
    "bar_volume_ylabel": "Index Volume",
    "bar_dataset_xlabel": "{dataset}"
  }
}
```

## 2. Text content

The `content` object contains all user-adjustable visible text.  Currently it
controls the bar-chart labels:

| key | default | used by |
|---|---|---|
| `bar_memory_ylabel` | `Index Memory` | memory bar chart |
| `bar_build_time_ylabel` | `Index Build Time (s)` | build-time bar chart |
| `bar_volume_ylabel` | `Index Volume` | volume bar chart |
| `bar_dataset_xlabel` | `{dataset}` | x-axis label, `{dataset}` is replaced by the dataset name |

To expose additional strings in the future, add them to the JSON `content`
object and read them in the plotting script through:

```python
import paper_style as PS

label = PS.text_content("my_key", "default text")
```

## 3. Using an alternative configuration file

Use another configuration file without editing the repository default:

```bash
export FIGURE_TEXT_CONFIG=/path/to/my_text_style.json
bash render_all_figures.sh
```

The bar chart script also accepts a `--text-config PATH` option:

```bash
python 5_Plot/fig_full_bars.py --metric memory --layout both \
    --text-config /path/to/my_text_style.json
```

The final line-plot scripts also accept the same option directly:

```bash
python 5_Plot/fig_final_sl_latency.py --text-config /path/to/my_text_style.json
python 5_Plot/fig_final_cp_latency.py --text-config /path/to/my_text_style.json
python 5_Plot/fig_final_latency_at_recall.py --text-config /path/to/my_text_style.json
```