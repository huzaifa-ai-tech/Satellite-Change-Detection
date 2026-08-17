import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.colors import HexColor

logger = logging.getLogger("satellite.pdf")

PRIMARY = HexColor("#0f172a")
ACCENT = HexColor("#06b6d4")
ACCENT_LIGHT = HexColor("#0891b2")
DARK_CARD = HexColor("#1e293b")
SUBTLE = HexColor("#94a3b8")
SUCCESS = HexColor("#10b981")
WARNING = HexColor("#f59e0b")
DANGER = HexColor("#ef4444")


class PDFReportGenerator:

    def _header_footer(self, canvas, doc):
        canvas.saveState()
        w, h = A4

        canvas.setFillColor(ACCENT)
        canvas.rect(0, h - 4, w, 4, fill=1, stroke=0)

        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(HexColor("#64748b"))
        canvas.drawString(50, 20, "Satellite AI Intelligence Report")
        canvas.drawRightString(w - 50, 20, f"Page {doc.page} | {datetime.now().strftime('%Y-%m-%d')}")

        canvas.rect(50, 28, w - 100, 0.5, fill=1, stroke=0)

        canvas.restoreState()

    def _cover_page(self, story, styles):
        w, h = A4

        spacer = 0
        story.append(Spacer(1, 2.5 * inch))

        s = ParagraphStyle("CoverAccent", parent=styles["Normal"], fontSize=11, textColor=ACCENT, alignment=1, fontName="Helvetica", spaceAfter=8)
        story.append(Paragraph("SATELLITE AI INTELLIGENCE REPORT", s))

        s2 = ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=32, leading=38, textColor=HexColor("#f8fafc"), alignment=1, spaceAfter=20)
        story.append(Paragraph("Change Detection<br/>&amp; Land Cover Analysis", s2))

        s3 = ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=12, leading=18, textColor=SUBTLE, alignment=1, spaceAfter=6)
        story.append(Paragraph("Automated satellite imagery analysis using ChangeFormerV6 + LoveDA SegFormer B2", s3))
        story.append(Paragraph("Powered by YOLO26-OBB vehicle detection (DOTA)", s3))

        story.append(Spacer(1, 1.5 * inch))

        info_style = ParagraphStyle("Info", parent=styles["Normal"], fontSize=9, textColor=SUBTLE, alignment=1, leading=16)
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", info_style))
        story.append(Paragraph("Version 2.0 &mdash; Professional Edition", info_style))

        story.append(PageBreak())

    def _summary_page(self, story, styles, change_percentage, obj_count, processing_time):
        story.append(Spacer(1, 24))

        s = ParagraphStyle("SectionTitle", parent=styles["Heading1"], fontSize=18, textColor=ACCENT, spaceAfter=4, fontName="Helvetica-Bold")
        story.append(Paragraph("Executive Summary", s))

        line = HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=16)
        story.append(line)

        summary_data = [
            ["Metric", "Value"],
            ["Changed Area", f"{change_percentage:.2f}%"],
            ["Objects Detected", str(obj_count)],
            ["Processing Time", f"{processing_time:.2f}s"],
            ["Model", "ChangeFormerV6 + LoveDA B2"],
            ["Vehicle Detector", "YOLO26-OBB (DOTA)"],
        ]
        t = Table(summary_data, colWidths=[3*inch, 3*inch], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 1), (-1, -1), DARK_CARD),
            ("TEXTCOLOR", (0, 1), (-1, -1), HexColor("#e2e8f0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LEFTPADDING", (0, 0), (-1, -1), 16),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#334155")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_CARD, HexColor("#162032")]),
        ]))
        story.append(t)
        story.append(PageBreak())

    def _image_section(self, story, styles, title, num, image_path):
        story.append(Spacer(1, 16))
        s = ParagraphStyle(f"H{num}", parent=styles["Heading2"], fontSize=14, textColor=ACCENT, spaceAfter=4)
        story.append(Paragraph(f"{num}. {title}", s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT_LIGHT, spaceAfter=12))

        if image_path and Path(image_path).exists():
            img = Image(image_path)
            img._restrictSize(5.5 * inch, 5 * inch)
            story.append(img)
        else:
            ns = ParagraphStyle("NA", parent=styles["Normal"], textColor=SUBTLE, alignment=1)
            story.append(Paragraph("Image not available", ns))

    def _objects_table(self, story, styles, objects):
        story.append(Spacer(1, 16))
        s = ParagraphStyle("ObjTitle", parent=styles["Heading2"], fontSize=14, textColor=ACCENT, spaceAfter=4)
        story.append(Paragraph("Detected Objects", s))
        story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT_LIGHT, spaceAfter=12))

        header_style = ParagraphStyle("Th", parent=styles["Normal"], textColor=colors.white, fontSize=9, fontName="Helvetica-Bold", alignment=1)
        cell_style = ParagraphStyle("Td", parent=styles["Normal"], textColor=HexColor("#e2e8f0"), fontSize=8.5, alignment=1)

        table_data = [
            [Paragraph("ID", header_style), Paragraph("Class", header_style),
             Paragraph("Pixels", header_style), Paragraph("Confidence", header_style),
             Paragraph("Status", header_style)]
        ]
        for obj in objects[:50]:
            table_data.append([
                Paragraph(str(obj.get("id", "")), cell_style),
                Paragraph(str(obj.get("class_name", "")), cell_style),
                Paragraph(str(obj.get("pixels", "")), cell_style),
                Paragraph(str(obj.get("confidence", "")), cell_style),
                Paragraph(str(obj.get("status", "")), cell_style),
            ])

        if len(table_data) == 1:
            table_data.append([Paragraph("-", cell_style)] * 5)

        t = Table(table_data, colWidths=[0.5*inch, 1.4*inch, 0.8*inch, 0.9*inch, 0.7*inch], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
            ("BACKGROUND", (0, 1), (-1, -1), DARK_CARD),
            ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#334155")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_CARD, HexColor("#162032")]),
        ]))
        story.append(t)

        if len(objects) > 50:
            ns = ParagraphStyle("More", parent=styles["Normal"], textColor=SUBTLE, fontSize=8, alignment=1, spaceAbove=6)
            story.append(Paragraph(f"... and {len(objects) - 50} more objects", ns))

    def create(self, output_path, before_image, after_image, overlay_image, change_percentage, statistics, objects, before_semantic=None, after_semantic=None, change_mask=None, chart_path=None, processing_time=0):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path), pagesize=A4,
            leftMargin=50, rightMargin=50, topMargin=50, bottomMargin=50,
        )
        styles = getSampleStyleSheet()

        s_normal = ParagraphStyle("DarkNormal", parent=styles["Normal"], textColor=HexColor("#e2e8f0"), fontSize=9.5)
        s_heading1 = ParagraphStyle("DarkH1", parent=styles["Heading1"], textColor=ACCENT, fontSize=18, spaceAfter=4)
        s_heading2 = ParagraphStyle("DarkH2", parent=styles["Heading2"], textColor=ACCENT, fontSize=14, spaceAfter=4)

        styles.add(s_normal)
        styles.add(s_heading1)
        styles.add(s_heading2)

        story = []

        self._cover_page(story, styles)
        self._summary_page(story, styles, change_percentage, len(objects), processing_time)

        self._image_section(story, styles, "Before Satellite Image", 1, before_image)
        story.append(PageBreak())
        self._image_section(story, styles, "After Satellite Image", 2, after_image)
        story.append(PageBreak())
        self._image_section(story, styles, "Binary Change Mask", 3, change_mask)
        story.append(PageBreak())
        self._image_section(story, styles, "Detection Overlay", 4, overlay_image)
        story.append(PageBreak())
        self._image_section(story, styles, "Object Distribution", 5, chart_path)
        story.append(PageBreak())

        self._objects_table(story, styles, objects)
        story.append(PageBreak())

        transitions = statistics.get("transitions", []) if isinstance(statistics, dict) else []
        if transitions:
            story.append(Spacer(1, 16))
            s = ParagraphStyle("TransTitle", parent=styles["Heading2"], fontSize=14, textColor=ACCENT, spaceAfter=4)
            story.append(Paragraph("Land Cover Transition Matrix", s))
            story.append(HRFlowable(width="100%", thickness=0.5, color=ACCENT_LIGHT, spaceAfter=12))

            hdr_s = ParagraphStyle("Th", parent=styles["Normal"], textColor=colors.white, fontSize=8.5, fontName="Helvetica-Bold", alignment=1)
            cel_s = ParagraphStyle("Td", parent=styles["Normal"], textColor=HexColor("#e2e8f0"), fontSize=8, alignment=1)
            tdata = [[Paragraph(h, hdr_s) for h in ["From", "To", "Pixels", "Percentage", "Severity"]]]
            for item in transitions[:30]:
                tdata.append([Paragraph(str(item.get(k, "")), cel_s) for k in ["from", "to", "pixels", "percentage", "severity"]])
            t = Table(tdata, colWidths=[1.2*inch, 1.2*inch, 1*inch, 0.9*inch, 1*inch], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("BACKGROUND", (0, 1), (-1, -1), DARK_CARD),
                ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#334155")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [DARK_CARD, HexColor("#162032")]),
            ]))
            story.append(t)

        doc.build(story, onFirstPage=self._header_footer, onLaterPages=self._header_footer)
        logger.info("PDF saved: %s", output_path)
        return str(output_path)
