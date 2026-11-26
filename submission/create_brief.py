import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "submission" / "validation_metrics.json"
OUTPUT_PATH = ROOT / "submission" / "Code-Blooded-VARSHA-Brief.pdf"
PAGE_WIDTH, PAGE_HEIGHT = 595, 842
NAVY = (0.055, 0.11, 0.20)
INK = (0.10, 0.16, 0.24)
MUTED = (0.34, 0.40, 0.49)
BLUE = (0.09, 0.39, 0.72)
TEAL = (0.03, 0.55, 0.52)
PALE = (0.93, 0.96, 0.98)
WHITE = (1.0, 1.0, 1.0)


class Page:
    def __init__(self):
        self.ops = []

    @staticmethod
    def _color(color):
        return " ".join(f"{component:.3f}" for component in color)

    def rect(self, x, y, width, height, color):
        self.ops.append(
            f"{self._color(color)} rg {x} {y} {width} {height} re f"
        )

    def line(self, x1, y1, x2, y2, color=MUTED, width=0.8):
        self.ops.append(
            f"{self._color(color)} RG {width:.2f} w {x1} {y1} m {x2} {y2} l S"
        )

    def text(self, x, y, value, size=10, *, bold=False, color=INK):
        escaped = str(value).encode("latin-1", "replace").decode("latin-1")
        escaped = escaped.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        font = "F2" if bold else "F1"
        self.ops.append(
            f"{self._color(color)} rg BT /{font} {size} Tf "
            f"1 0 0 1 {x} {y} Tm ({escaped}) Tj ET"
        )

    def wrapped(self, x, y, value, width_chars=82, size=9, leading=13, *, color=INK):
        lines = textwrap.wrap(str(value), width=width_chars, break_long_words=False)
        for line in lines:
            self.text(x, y, line, size, color=color)
            y -= leading
        return y

    def section(self, x, y, title, *, color=BLUE):
        self.rect(x, y - 3, 5, 16, color)
        self.text(x + 12, y, title.upper(), 10, bold=True, color=NAVY)

    def finish(self):
        return ("\n".join(self.ops) + "\n").encode("latin-1")


def _banner(page, page_number, subtitle):
    page.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, PALE)
    page.rect(0, PAGE_HEIGHT - 130, PAGE_WIDTH, 130, NAVY)
    page.text(42, 789, "CODE BLOODED  /  VARSHA", 10, bold=True, color=(0.45, 0.84, 0.78))
    page.text(42, 751, "Multiverse of Monsoon", 25, bold=True, color=WHITE)
    page.text(42, 724, subtitle, 10, color=(0.82, 0.88, 0.94))
    page.text(42, 28, "Offline-first climate simulation | India 32 x 36 daily grid", 8, color=MUTED)
    page.text(541, 28, f"{page_number} / 2", 8, bold=True, color=BLUE)


def _card(page, x, y, width, height, title, body, accent=BLUE):
    page.rect(x, y - height, width, height, WHITE)
    page.rect(x, y - 4, width, 4, accent)
    page.text(x + 12, y - 24, title, 10, bold=True, color=NAVY)
    page.wrapped(x + 12, y - 43, body, width_chars=max(20, int(width / 5.3)), size=8, leading=11, color=MUTED)


def _validation_summary():
    if not METRICS_PATH.is_file():
        return (
            "Run training/evaluation before finalizing this brief. No benchmark result "
            "is claimed until the checked-in weights have been evaluated."
        )
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    leads = metrics.get("leads", {})
    pieces = []
    for lead in ("1", "4", "8"):
        record = leads.get(lead)
        if not record or "skill_vs_best_baseline" not in record:
            continue
        skills = record["skill_vs_best_baseline"]
        mean_skill = sum(skills) / len(skills)
        csi = record.get("rain_csi_10mm_day", {}).get("model")
        csi_text = f"; rain CSI {csi:.2f}" if csi is not None else ""
        pieces.append(f"+{lead}d mean skill {mean_skill:+.0%}{csi_text}")
    ratios = metrics.get("day8_spatial_variance_ratio")
    spread = (
        f"Day-8 spatial variance ratios range {min(ratios):.2f}-{max(ratios):.2f}."
        if ratios
        else ""
    )
    if not pieces:
        return "Metrics file has no model results. Re-run the evaluator for the selected checkpoint."
    training_years = metrics.get("checkpoint_training_years", "unrecorded")
    if training_years == "unrecorded":
        provenance = "Checkpoint training years are unrecorded; 2021 holdout validity is unverified."
    elif isinstance(training_years, list) and 2021 in training_years:
        provenance = "Checkpoint includes 2021; these results are not holdout evidence."
    else:
        provenance = "Checkpoint metadata records training years excluding 2021."
    rain_skill = leads.get("8", {}).get("skill_vs_best_baseline", [])
    if len(rain_skill) > 1:
        pieces.append(f"day-8 rain skill {rain_skill[1]:+.0%}")
    rain_mode = metrics.get("checkpoint_rain_mode")
    rain_note = (
        " The selected legacy rain head only predicts nonnegative increments."
        if rain_mode == "softplus_increment"
        else ""
    )
    return (
        f"2021 comparison, {metrics['windows']} windows: "
        + "; ".join(pieces)
        + ". "
        + spread
        + " "
        + provenance
        + rain_note
        + " This does not establish skill on hidden 2023-24 cases."
    )


def build_pages():
    first = Page()
    _banner(
        first,
        1,
        "MODEL DESIGN  |  AUTOREGRESSIVE ROLLOUT, STABILITY, VALIDATION",
    )
    first.section(42, 687, "A learned daily state transition")
    first.wrapped(
        42,
        665,
        "VARSHA is a compact six-block, 64-channel residual convolutional model. It receives three consecutive "
        "physical weather states normalized with the kit statistics, two bundled geography "
        "fields, and calendar-aligned day-of-year sine/cosine planes. The network predicts "
        "normalized next-day changes; the server applies bounds and feeds the new state "
        "back for the next forecast day.",
        width_chars=98,
        size=9,
        leading=13,
    )

    card_y, card_h, card_w, gap = 603, 82, 119, 10
    cards = (
        ("01  CONTEXT", "3 days x 6 channels\nPhysical-unit API input", BLUE),
        ("02  CONDITION", "Static terrain + land/sea\nSeasonal sine/cosine", TEAL),
        ("03  TRANSITION", "Residual CNN predicts\nthe next-day increment", BLUE),
        ("04  ROLLOUT", "Guard rain and temperature;\nrecycle state autoregressively", TEAL),
    )
    for index, (title, body, accent) in enumerate(cards):
        _card(first, 42 + index * (card_w + gap), card_y, card_w, card_h, title, body, accent)

    first.section(42, 492, "Drift controls and known failure modes")
    first.wrapped(
        42,
        469,
        "Channel normalization, the three-day context, seasonal phase, static geography, "
        "finite-value checks, a nonnegative-rain floor, and -40 C to +60 C temperature "
        "bounds protect the rollout from common numerical failures. Training now includes "
        "multi-day autoregressive loss, so later model inputs include the model's own "
        "predictions. These are engineering guards, not conservation laws.",
        width_chars=98,
        size=8.5,
        leading=12,
    )
    first.wrapped(
        42,
        397,
        "Limits: deterministic 1-degree daily fields cannot resolve local storms or "
        "sub-daily extremes. Error compounds with lead time; rainfall and climate trends "
        "can be misrepresented. The 2005-2021 data cannot establish skill on the hidden "
        "2023-2024 cases. Rain, flood, and wind-energy indicators are screening proxies, "
        "not operational warnings or calibrated impact forecasts.",
        width_chars=98,
        size=8.5,
        leading=12,
        color=MUTED,
    )

    first.section(42, 322, "Temporal holdout diagnostics")
    first.rect(42, 180, 511, 118, WHITE)
    first.wrapped(
        56,
        275,
        _validation_summary(),
        width_chars=102,
        size=8.5,
        leading=13,
    )
    first.wrapped(
        56,
        216,
        "Evaluation compares identical 2021 target windows against both persistence and "
        "training-period day-of-year climatology, with latitude-weighted RMSE and a 10 "
        "mm/day heavy-rain CSI. Check validation_metrics.json and rerun the documented "
        "command whenever model weights change.",
        width_chars=102,
        size=8,
        leading=12,
        color=MUTED,
    )

    second = Page()
    _banner(
        second,
        2,
        "LIVE VISUALIZATION  |  COUNTERFACTUAL SHOCKS  |  IMPACT SCREENING",
    )
    second.section(42, 687, "A live, local multiverse")
    second.wrapped(
        42,
        665,
        "The separate Streamlit dashboard selects a real three-day state window from the "
        "team-kit NetCDF, sends it to the local simulation API, and plots the returned "
        "32 x 36 fields. Baseline and alternate forecasts use a shared color scale; a "
        "separate map shows their physical-unit difference. Nothing is pre-rendered and "
        "no remote weather service is contacted.",
        width_chars=98,
        size=9,
        leading=13,
    )

    map_x, map_y = 48, 475
    cell = 12
    for row in range(13):
        for col in range(18):
            ratio = (row / 12 + col / 17) / 2
            red = min(1.0, 0.15 + 0.80 * ratio)
            blue = min(1.0, 0.84 - 0.68 * ratio)
            green = min(1.0, 0.42 + 0.12 * (1 - abs(ratio - 0.5) * 2))
            second.rect(
                map_x + col * cell,
                map_y - row * cell,
                cell - 1,
                cell - 1,
                (red, green, blue),
            )
    second.text(48, 295, "Illustrative grid layout; dashboard maps are generated live.", 7.5, color=MUTED)
    _card(
        second,
        300,
        472,
        253,
        95,
        "PHYSICAL SHOCKS",
        "Warm Bay: +3 K and +15% TCWV in the kit box. Arabian Sea stall: -40% u10 and v10. Zero shock is an exact identity check.",
        TEAL,
    )
    _card(
        second,
        300,
        362,
        253,
        95,
        "IMPACT SCREEN",
        "Area-weighted rain deficits, hot-area-days above 35 C, heavy-rain coverage, peak rain, and wind-speed-cubed index.",
        BLUE,
    )

    second.section(42, 268, "Impact use case: monsoon stress and heat screening")
    second.wrapped(
        42,
        245,
        "A stalled monsoon scenario can expose how a short-range rainfall deficit changes "
        "across central India. Latitude-weighted rain deficits, heavy-rain coverage, and "
        "a hot-area-day proxy above 35 C let viewers compare the baseline and alternate "
        "branch when discussing crop-water stress, heat exposure, or flood preparedness.",
        width_chars=98,
        size=8.5,
        leading=12,
    )
    second.wrapped(
        42,
        190,
        "The indicators are deliberately labeled as screening proxies: there is no crop "
        "calendar, soil moisture, irrigation, river routing, exposure data, or turbine "
        "calibration in the supplied kit. They must not be interpreted as yield, flood "
        "depth, or electricity forecasts.",
        width_chars=98,
        size=8.5,
        leading=12,
        color=MUTED,
    )

    second.section(42, 120, "Offline and reproducible")
    second.wrapped(
        42,
        97,
        "Inference uses only the bundled model, statistics, and static map. The dashboard "
        "restricts API calls to localhost; no internet or weather API is used. Training "
        "reads the kit data locally, while the repository ships weights rather than a "
        "multi-year archive. See README.md for commands and limitations.",
        width_chars=98,
        size=8.3,
        leading=12,
    )
    return [first.finish(), second.finish()]


def _pdf_bytes(page_contents):
    objects = [None]

    def add_object(value):
        objects.append(value)
        return len(objects) - 1

    font_regular = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_bold = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    pages_id = add_object(None)
    page_ids = []
    for content in page_contents:
        content_id = add_object(
            b"<< /Length "
            + str(len(content)).encode("ascii")
            + b" >>\nstream\n"
            + content
            + b"endstream"
        )
        page_id = add_object(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    )
    catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects[1:], start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects)}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects)} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(_pdf_bytes(build_pages()))
    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
