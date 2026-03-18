import os
import subprocess
import sys
import time
from pathlib import Path

import uno


HOST = "127.0.0.1"
PORT = 2002


SLIDES = [
    {
        "title": "AURA Project Summary",
        "bullets": [
            "AURA turns live robot video into simple, usable guidance.",
            "It watches the scene, highlights risks, and suggests the next safe move.",
            "The goal is clarity for operators, not technical complexity.",
        ],
    },
    {
        "title": "What The System Does",
        "bullets": [
            "Accepts live camera or recorded video as input.",
            "Describes what is happening around the robot in real time.",
            "Shows clear movement suggestions such as move forward, turn, or hold.",
            "Keeps a visible history so users can review what the system saw.",
        ],
    },
    {
        "title": "Key Improvements Completed",
        "bullets": [
            "Fixed synchronization so displayed results better match the viewed image.",
            "Added multi-image reasoning so the system can use several recent frames together.",
            "Enabled the same multi-frame support for both video files and live camera mode.",
            "Adjusted speech alerts so the system speaks only when a hazard is detected.",
            "Integrated MLflow so runs, outputs, and warnings can be tracked over time.",
        ],
    },
    {
        "title": "Why These Changes Matter",
        "bullets": [
            "Operators get more trustworthy guidance.",
            "Warnings are less noisy and easier to pay attention to.",
            "Movement suggestions are based on a short sequence, not a single snapshot.",
            "The system is easier to explain in demos and evaluations.",
        ],
    },
    {
        "title": "Recent Reliability Upgrade",
        "bullets": [
            "Each result now carries a frame reference and capture time.",
            "This makes it easier to confirm that the decision came from the correct moment.",
            "It also improves review, troubleshooting, and presentation quality.",
        ],
    },
    {
        "title": "Current Value",
        "bullets": [
            "Safer and clearer robot awareness for indoor navigation scenarios.",
            "Simple user interface with real-time captions, decisions, and hazard alerts.",
            "MLflow records results across runs, making comparisons and reporting easier.",
            "A stronger base for demos, academic reporting, and future field testing.",
        ],
    },
    {
        "title": "How MLflow Helps",
        "bullets": [
            "Keeps a record of what the system saw and what it decided.",
            "Makes it easier to compare different prompts, settings, and models.",
            "Supports project reporting with saved outputs and run history.",
            "Helps the team review progress without reading technical logs.",
        ],
    },
    {
        "title": "Recommended Next Steps",
        "bullets": [
            "Improve motion understanding across several frames even further.",
            "Use confidence levels to make the robot more conservative when unsure.",
            "Present decisions with a short plain-language explanation for non-technical users.",
            "Expand evaluation with more realistic robot paths and edge cases.",
        ],
    },
]


def start_office():
    cmd = [
        "libreoffice",
        "--headless",
        "--nologo",
        "--nodefault",
        "--norestore",
        f'--accept=socket,host={HOST},port={PORT};urp;',
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def connect():
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx
    )
    url = f"uno:socket,host={HOST},port={PORT};urp;StarOffice.ComponentContext"
    last_err = None
    for _ in range(50):
        try:
            return resolver.resolve(url)
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(0.2)
    raise RuntimeError(f"Could not connect to LibreOffice: {last_err}")


def make_property(name, value):
    prop = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    prop.Name = name
    prop.Value = value
    return prop


def set_text(shape, text):
    shape.setString(text)
    cursor = shape.createTextCursor()
    cursor.gotoStart(False)
    cursor.gotoEnd(True)


def create_textbox(doc, slide, x, y, w, h, text, font_size, bold=False):
    shape = doc.createInstance("com.sun.star.drawing.TextShape")
    shape.setPosition(uno.createUnoStruct("com.sun.star.awt.Point", x, y))
    shape.setSize(uno.createUnoStruct("com.sun.star.awt.Size", w, h))
    slide.add(shape)
    set_text(shape, text)
    cursor = shape.createTextCursor()
    cursor.CharHeight = font_size
    cursor.CharFontName = "Liberation Sans"
    if bold:
        cursor.CharWeight = 150.0
    return shape


def clear_slide(slide):
    while slide.getCount() > 0:
        slide.remove(slide.getByIndex(0))


def add_slide(doc, slide, title, bullets):
    clear_slide(slide)
    create_textbox(doc, slide, 800, 700, 24000, 1800, title, 28, bold=True)

    bullet_text = "\n".join([f"• {line}" for line in bullets])
    body = create_textbox(doc, slide, 1200, 3200, 22000, 9000, bullet_text, 20)
    body.TextAutoGrowHeight = True
    return slide


def build_presentation(out_path: Path):
    office = start_office()
    try:
        ctx = connect()
        smgr = ctx.ServiceManager
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.loadComponentFromURL(
            "private:factory/simpress", "_blank", 0, ()
        )

        pages = doc.getDrawPages()
        if pages.getCount() == 0:
            pages.insertNewByIndex(0)

        first = pages.getByIndex(0)
        add_slide(doc, first, SLIDES[0]["title"], SLIDES[0]["bullets"])
        for slide_data in SLIDES[1:]:
            pages.insertNewByIndex(pages.getCount())
            slide = pages.getByIndex(pages.getCount() - 1)
            add_slide(doc, slide, slide_data["title"], slide_data["bullets"])

        out_url = uno.systemPathToFileUrl(str(out_path))
        props = (
            make_property("FilterName", "Impress MS PowerPoint 2007 XML"),
        )
        doc.storeAsURL(out_url, props)
        doc.close(True)
    finally:
        office.terminate()
        office.wait(timeout=10)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "AURA_Project_Summary.pptx"
    build_presentation(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
