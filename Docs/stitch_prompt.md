# Stitch prompt — HDFC Fund Facts Assistant UI

Paste this prompt into [Google Stitch](https://stitch.withgoogle.com) to
generate the UI design. It pins **Groww's brand colors** and mirrors the
one-screen layout the Streamlit app (`ui/app.py`) already implements, so
the generated design can act as the visual source of truth.

Download the result as `stitch_hdfc_fund_facts_assistant.zip` and place it
in the project root (`E:\Ragchatbot`).

---

## Prompt (copy everything below the line)

Design a single-page web UI for "HDFC Fund Facts Assistant" — a facts-only
Q&A assistant for HDFC mutual fund schemes (corpus: one Groww AMC page +
four scheme pages). Professional fintech look, clean and calm, generous
whitespace, rounded cards (8px), subtle shadows, light theme only,
responsive desktop-first. Font: Inter (system fallback). Use ONLY the
Groww brand palette:

- Primary green (buttons, links, accents, logo): #00D09C
- Primary green hover / pressed: #00B386
- Headline / body ink: #44475B
- Secondary muted text: #7E838D
- Page background: #FFFFFF, section background #F5F6F8
- Borders and dividers: #E2E4EB
- Danger / refusal accents: #EC5B5B
- Success / fact highlights: #00D09C at 12% opacity

Structure, top to bottom:

1. Sticky header — white, bottom border #E2E4EB. Left: a 28px rounded
   square logo mark in #00D09C with a white rising-bar chart glyph, then
   the app name "HDFC Fund Facts Assistant" in #44475B, 16px semibold.
   Right: a subtle pill badge "Facts-only. No investment advice." in
   #7E838D, 12px, bordered pill.
2. Hero section (centered, max width 720px) — headline "Ask a factual
   question about HDFC mutual funds" (28px, #44475B), subline
   "Expense ratio, exit load, minimum SIP, lock-in, riskometer,
   benchmark — every answer cites one official Groww page."
   (15px, #7E838D).
3. Question card — white card, 1px #E2E4EB border, 8px radius:
   - A text input (placeholder "e.g. What is the expense ratio of HDFC
     Large Cap?") with a green #00D09C "Ask" button on the right.
   - Below the input, three example chips (pill buttons, #F5F6F8 fill,
     #44475B text, 1px #E2E4EB border): "Flexi Cap expense ratio",
     "ELSS lock-in", "Large Cap exit load".
4. Answer card — white card, 8px radius, left accent bar 4px #00D09C:
   - Answer text: up to 3 short sentences, 16px, #44475B, line height 1.6.
   - Citation row: "Source: HDFC ELSS Tax Saver Fund ↗" as a #00D09C
     text link with an external-link icon, and below it a muted caption
     "Last updated from sources: 11 Nov 2026" (12px, #7E838D).
   - A collapsible "Show retrieval details" row (chevron icon, #7E838D)
     that expands to a small table: query id, intent, path, filter, and
     up to 6 retrieved chunks with monospace chunk ids, similarity
     scores, section and fact-key columns (12px, zebra rows #F5F6F8).
5. A second, dimmed example card showing a guardrail refusal: text "I can
   only share factual scheme details, so I can't advise whether this fund
   suits you." with a #EC5B5B left accent bar and the same citation row
   style.
6. Footer — centered, 13px, #7E838D: "Facts-only. No investment advice.
   Corpus: 5 official Groww pages. Personal details are never stored."

Accessibility: text contrast at least 4.5:1 (do not put white text on
#00D09C — use #44475B text with green accents instead), focus rings
visible on the input and chips, 44px tap targets for buttons.
