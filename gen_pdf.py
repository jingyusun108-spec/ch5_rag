# -*- coding: utf-8 -*-
"""生成示例 PDF 知识库文件（供 PDF 加载器与问答系统使用）。

使用 fpdf2 + simhei 中文字体，将 knowledge_base/rag_intro.txt 转为 PDF。
运行：python gen_pdf.py
"""
import os
from fpdf import FPDF

BASE = os.path.dirname(os.path.abspath(__file__))
TXT = os.path.join(BASE, "knowledge_base", "rag_intro.txt")
PDF = os.path.join(BASE, "knowledge_base", "rag_intro.pdf")
FONT = r"C:\Windows\Fonts\simhei.ttf"


class ChinesePDF(FPDF):
    def header(self):
        self.set_font("simhei", size=10)
        self.cell(0, 10, "农业种植技术手册", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


def main():
    with open(TXT, encoding="utf-8") as f:
        lines = f.read().splitlines()

    pdf = ChinesePDF()
    pdf.add_font("simhei", "", FONT)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            pdf.set_font("simhei", size=14)
            pdf.cell(0, 8, line.lstrip("# "), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("## "):
            pdf.set_font("simhei", size=12)
            pdf.cell(0, 8, line.lstrip("# "), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("simhei", size=10)
            pdf.multi_cell(0, 7, line)
        pdf.ln(2)

    pdf.output(PDF)
    print(f"已生成：{PDF}  ({os.path.getsize(PDF)} bytes)")


if __name__ == "__main__":
    main()
