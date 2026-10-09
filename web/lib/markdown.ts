/** Normalize dollar math outside code while keeping money and shell variables literal. */
function normalizeText(text: string): string {
  let out = "";
  for (let i = 0; i < text.length;) {
    if (text[i] === "\\") {
      out += text.slice(i, i + 2);
      i += 2;
      continue;
    }
    if (text[i] !== "$") {
      out += text[i++];
      continue;
    }
    if (text.startsWith("$$", i)) {
      const end = text.indexOf("$$", i + 2);
      if (end >= 0 && text.slice(i + 2, end).trim()) {
        out += `\n$$\n${text.slice(i + 2, end).trim()}\n$$\n`;
        i = end + 2;
        continue;
      }
    } else {
      let end = i + 1;
      while (end < text.length && text[end] !== "\n") {
        if (text[end] === "\\") { end += 2; continue; }
        if (text[end] === "$") break;
        end++;
      }
      const body = text.slice(i + 1, end);
      // Delimiter-adjacent whitespace or a following word signals ordinary dollars.
      if (text[end] === "$" && body && body.trim() === body && !/[\w$]/.test(text[end + 1] || "")) {
        out += text.slice(i, end + 1);
        i = end + 1;
        continue;
      }
    }
    out += "\\$";
    i++;
  }
  return out;
}

export function sanitizeMathDollars(text: string): string {
  let result = "";
  let offset = 0;
  // Code samples are literal, including dollars and backslashes inside them.
  for (const match of text.matchAll(/(`+)[\s\S]*?\1|(~{3,})[\s\S]*?\2/g)) {
    result += normalizeText(text.slice(offset, match.index)) + match[0];
    offset = match.index! + match[0].length;
  }
  return result + normalizeText(text.slice(offset));
}

/** Link citations only in prose, preserving array indices in formulas and code. */
export function linkReportCitations(text: string): string {
  return text.split(/(```[\s\S]*?```|`[^`]*`|\$\$[\s\S]*?\$\$|(?<!\\)\$[^\n$]+?\$)/g)
    .map((part, index) => index % 2 ? part : part.replace(/(?<!!)\[(\d+)\](?!\()/g, '[$1](#source-$1)'))
    .join("");
}
