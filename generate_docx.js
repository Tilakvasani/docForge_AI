/**
 * DocForge AI — generate_docx.js  v2.0
 * - Renders pipe-formatted tables as real Word tables
 * - Plain text sections as paragraphs
 * - Professional styling: header, footer, page numbers
 *
 * Usage: node generate_docx.js <input.json> <output.docx>
 */

const fs   = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel,
  BorderStyle, WidthType, ShadingType, VerticalAlign,
  PageNumber, PageBreak,
} = require('docx');

const [,, inputPath, outputPath] = process.argv;
if (!inputPath || !outputPath) {
  console.error('Usage: node generate_docx.js <input.json> <output.docx>');
  process.exit(1);
}

const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const { doc_type, department, company_name, industry, region, sections } = input;

// ── Helpers ────────────────────────────────────────────────────────────────

const ACCENT  = "2E4057";
const GRAY    = "888888";
const LGRAY   = "F3F4F6";

function isTableRow(line)      { return line.includes('|'); }
function isSeparatorRow(line)  { return /^\s*[\|\-\s:]+$/.test(line) && line.includes('-'); }

function parseTableLines(lines) {
  // Returns array of rows, each row is array of cell strings
  return lines
    .filter(l => isTableRow(l) && !isSeparatorRow(l))
    .map(l => l.split('|').map(c => c.trim()).filter((_, i, arr) => i > 0 && i < arr.length - 1 || (arr[0] !== '' || arr[arr.length-1] !== '')))
    .map(row => {
      // Handle leading/trailing empty from "| a | b |" format
      const cells = l => l.split('|').map(c => c.trim());
      return cells(lines.find(x => x === lines[lines.indexOf(l)] ));
    });
}

// Better parser
function parseTable(tableLines) {
  const rows = [];
  for (const line of tableLines) {
    if (isSeparatorRow(line)) continue;
    if (!isTableRow(line)) continue;
    const raw = line.split('|');
    // Remove empty first/last if the line starts/ends with |
    const cells = raw.map(c => c.trim());
    const clean = (cells[0] === '' ? cells.slice(1) : cells);
    const final = (clean[clean.length-1] === '' ? clean.slice(0,-1) : clean);
    if (final.length > 0) rows.push(final);
  }
  return rows;
}

function makeWordTable(rows) {
  if (!rows || rows.length === 0) return null;

  const colCount = Math.max(...rows.map(r => r.length));
  const tableW   = 9360;
  const colW     = Math.floor(tableW / colCount);
  const colWidths = Array(colCount).fill(colW);

  const border = { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC" };
  const borders = { top: border, bottom: border, left: border, right: border };

  const tableRows = rows.map((row, ri) => {
    const isHeader = ri === 0;
    return new TableRow({
      tableHeader: isHeader,
      children: Array.from({ length: colCount }, (_, ci) => {
        const cellText = row[ci] || '';
        return new TableCell({
          borders,
          width: { size: colW, type: WidthType.DXA },
          shading: isHeader
            ? { fill: ACCENT, type: ShadingType.CLEAR }
            : (ri % 2 === 0 ? { fill: LGRAY, type: ShadingType.CLEAR } : undefined),
          margins: { top: 80, bottom: 80, left: 120, right: 120 },
          children: [new Paragraph({
            spacing: { after: 0 },
            children: [new TextRun({
              text: cellText,
              font: "Arial",
              size: 20,
              bold: isHeader,
              color: isHeader ? "FFFFFF" : "222222",
            })],
          })],
        });
      }),
    });
  });

  return new Table({
    width: { size: tableW, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: tableRows,
  });
}

// ── Parse section content into blocks ─────────────────────────────────────

function parseSectionToBlocks(content) {
  const blocks = [];
  const rawLines = content.split('\n');

  let i = 0;
  while (i < rawLines.length) {
    const line = rawLines[i];

    // Collect a table block
    if (isTableRow(line)) {
      const tableLines = [];
      while (i < rawLines.length && (isTableRow(rawLines[i]) || isSeparatorRow(rawLines[i]))) {
        tableLines.push(rawLines[i]);
        i++;
      }
      const rows = parseTable(tableLines);
      if (rows.length > 0) {
        const tbl = makeWordTable(rows);
        if (tbl) {
          blocks.push(tbl);
          blocks.push(new Paragraph({ spacing: { after: 120 } })); // spacer after table
        }
      }
      continue;
    }

    const trimmed = line.trim();

    if (!trimmed) {
      blocks.push(new Paragraph({ spacing: { after: 80 } }));
      i++; continue;
    }

    // Numbered: "1. Text"
    const numMatch = trimmed.match(/^(\d+)[.)]\s+(.+)$/);
    if (numMatch) {
      blocks.push(new Paragraph({
        spacing: { after: 80 },
        indent: { left: 360 },
        children: [new TextRun({ text: `${numMatch[1]}. ${numMatch[2]}`, font: "Arial", size: 22 })],
      }));
      i++; continue;
    }

    // Bullet: "- Text"
    const bulletMatch = trimmed.match(/^[-•]\s+(.+)$/);
    if (bulletMatch) {
      blocks.push(new Paragraph({
        spacing: { after: 80 },
        indent: { left: 360, hanging: 180 },
        children: [
          new TextRun({ text: "•  ", font: "Arial", size: 22 }),
          new TextRun({ text: bulletMatch[1], font: "Arial", size: 22 }),
        ],
      }));
      i++; continue;
    }

    // Regular text
    blocks.push(new Paragraph({
      spacing: { after: 120 },
      children: [new TextRun({ text: trimmed, font: "Arial", size: 22 })],
    }));
    i++;
  }

  return blocks;
}

// ── Build document ─────────────────────────────────────────────────────────

const children = [];

// Title
children.push(new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { after: 200 },
  children: [new TextRun({ text: doc_type, font: "Arial", size: 40, bold: true, color: ACCENT })],
}));

// Subtitle line
children.push(new Paragraph({
  spacing: { after: 240 },
  children: [new TextRun({ text: `${company_name}  ·  ${department}  ·  ${region}`, font: "Arial", size: 20, color: GRAY })],
}));

// Divider
children.push(new Paragraph({
  border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT } },
  spacing: { after: 280 },
  children: [new TextRun({ text: "" })],
}));

// Sections
sections.forEach((sec, idx) => {
  if (idx > 0) children.push(new Paragraph({ spacing: { after: 160 } }));

  // Section heading
  children.push(new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text: sec.name, font: "Arial", size: 26, bold: true, color: ACCENT })],
  }));

  // Section content
  const contentBlocks = parseSectionToBlocks(sec.content || "");
  children.push(...contentBlocks);
});

// Footer note
children.push(new Paragraph({ spacing: { after: 400 } }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: `${doc_type}  ·  Generated by DocForge AI  ·  Confidential`,
    font: "Arial", size: 16, color: "AAAAAA",
  })],
}));

// ── Assemble ───────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 40, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 0, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: `${company_name}  |  ${doc_type}`, font: "Arial", size: 16, color: GRAY })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: "Arial", size: 16, color: GRAY }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: GRAY }),
            new TextRun({ text: " of ", font: "Arial", size: 16, color: GRAY }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 16, color: GRAY }),
          ],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outputPath, buf);
  console.log(`OK:${outputPath}`);
}).catch(err => {
  console.error('DOCX error:', err.message);
  process.exit(1);
});