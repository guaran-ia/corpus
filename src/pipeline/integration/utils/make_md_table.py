from statistics import mean

def make_markdown_table(
    rows: list[dict],
    *,
    rename: dict[str, str] | None = None,
    order_by: str | list[str] | list[tuple[str, bool]] | None = None,
    descending: bool = True,
    highlights: dict[str, str] | None = None,
    totals: dict[str, str] | None = None,
    formatters: dict[str, callable] | None = None,
    align: dict[str, str] | None = None,
) -> str:

    rename = rename or {}
    highlights = highlights or {}
    totals = totals or {}
    formatters = formatters or {}
    align = align or {}

    processed = []

    #Rename the columns
    for row in rows:
        new_row = {}
        for k, v in row.items():
            new_row[rename.get(k, k)] = v
        processed.append(new_row)

    columns = list(processed[0].keys())

    if isinstance(align, str):
        align = {col: align for col in columns}

    if order_by:
        # Normalize order_by into a list of (column, descending) tuples
        if isinstance(order_by, str):
            keys = [(order_by, descending)]
        elif isinstance(order_by, list):
            if isinstance(order_by[0], tuple):
                keys = order_by
            else:
                keys = [(col, descending) for col in order_by]
        else:
            raise TypeError("order_by must be str or list")

        # Stable sort: apply from lowest priority to highest
        for col, desc in reversed(keys):
            processed.sort(key=lambda r: r[col], reverse=desc)

    highlight_values = {}
    for col, mode in highlights.items():
        values = [r[col] for r in processed]
        highlight_values[col] = max(values) if mode == "max" else min(values)

    md = []
    md.append("| " + " | ".join(columns) + " |")

    def align_marker(col):
        a = align.get(col, "centre")
        return {
            "left": ":---",
            "right": "---:",
            "centre": ":---:",
        }[a]

    md.append("| " + " | ".join(align_marker(c) for c in columns) + " |")

    for row in processed:
        cells = []
        for col in columns:
            val = row[col]

            if col in formatters:
                val = formatters[col](val)
            else:
                val = str(val)

            if col in highlight_values and row[col] == highlight_values[col]:
                val = f"**{val}**"

            cells.append(val)

        md.append("| " + " | ".join(cells) + " |")

    if totals:
        total_cells = []
        for col in columns:
            if col in totals:
                rule = totals[col]

                if callable(rule):
                    val = rule(processed)

                else:
                    values = [r[col] for r in processed]
                    if rule == "sum":
                        val = sum(values)
                    elif rule == "mean":
                        val = mean(values)
                    else:
                        val = ""
                val = formatters[col](val) if col in formatters else str(val)
                total_cells.append(f"**{val}**")
            else:
                total_cells.append("")
        md.append("| " + " | ".join(total_cells) + " |")
    
    return "\n".join(md)