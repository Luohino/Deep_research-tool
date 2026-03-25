# Lutervyn Deep Research

Lutervyn Deep Research is a Python-based research assistant that searches the web, scrapes useful pages, extracts relevant information, and generates a readable HTML report.

The project is built around a single script, [run_custom_researcher.py](F:\Luohino\Lutervyn\Lutervyn Deep research\lutervyndr\deep_research\run_custom_researcher.py), which handles the full workflow from query generation to report export.

## What It Does

Given a topic or query, the tool:

- generates focused search queries using an LLM
- searches the web using DuckDuckGo, with Google as a fallback
- fetches and parses webpage content
- extracts useful facts and summaries from the scraped text
- decides whether additional research iterations are needed
- searches for relevant images
- builds a final HTML report with source links, clickable citations, social/profile links, and image previews

## Features

- Iterative research loop driven by LLM-generated search queries
- Async web fetching with `aiohttp`
- HTML parsing with `BeautifulSoup`
- DuckDuckGo web and image search
- Google fallback for web search
- Source tracking across the full run
- Clickable citations inside the final report
- Search trail section showing explored links
- Custom HTML report output stored locally in the `reports/` folder
- Soft light report theme that can be restyled through the embedded CSS

## Project Structure

```text
deep_research/
├── run_custom_researcher.py
├── .env
├── reports/
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- An API key for the LLM endpoint used by the script

## Python Dependencies

Install the required packages:

```bash
pip install aiohttp beautifulsoup4 duckduckgo-search googlesearch-python markdown nest_asyncio python-dotenv
```

If `pip` is not available directly, use:

```bash
python -m pip install aiohttp beautifulsoup4 duckduckgo-search googlesearch-python markdown nest_asyncio python-dotenv
```

## Environment Variables

The script looks for an API key in this order:

1. `NVIDIA_API_KEY`
2. `OPENROUTER_API_KEY`
3. `API_KEY`

You can create a `.env` file in the project root:

```env
NVIDIA_API_KEY=your_api_key_here
```

Or set one of the environment variables in your shell before running the script.

## How to Run

Run with command-line arguments:

```bash
python run_custom_researcher.py "your research topic here" 5
```

Arguments:

- first argument: the research query
- second argument: maximum number of iterations

Example:

```bash
python run_custom_researcher.py "who is luohino developer github twitter" 5
```

You can also run it interactively:

```bash
python run_custom_researcher.py
```

The script will ask for:

- the research query
- the maximum iteration count

## How the Workflow Works

### 1. Query Planning

The script sends the user query to the LLM and asks for a small set of focused search queries.

### 2. Web Search

Each generated query is searched using DuckDuckGo. If DuckDuckGo does not return usable results, the script falls back to Google search.

### 3. Page Fetching and Scraping

Each discovered URL is fetched asynchronously. The script extracts:

- page title
- meta description
- Open Graph image when available
- main visible text content

### 4. Context Extraction

The scraped text is passed to the LLM, which extracts the most relevant information for the original user query.

### 5. Iterative Refinement

After each batch of results, the tool asks the LLM whether more research is needed. If so, it generates another round of search queries.

### 6. Image Search

The script performs a separate image search and collects preview image URLs, titles, and source page links when available.

### 7. Final Report

At the end of the run, the tool generates:

- a final written report
- an HTML preview page
- cited source cards
- a search trail
- image preview cards
- social/profile links detected during the run

## Output

Generated reports are saved in the `reports/` directory.

The HTML report includes:

- report title and metadata
- summary and deep-dive content
- clickable citations
- source cards
- image preview cards
- social/profile links
- full cited source list
- explored link trail

## Customizing the HTML Theme

The HTML template is embedded directly inside [run_custom_researcher.py](F:\Luohino\Lutervyn\Lutervyn Deep research\lutervyndr\deep_research\run_custom_researcher.py).

If you want to change the appearance:

- edit the CSS variables inside `generate_html_report(...)`
- adjust background, border, accent, and text colors
- change spacing, card sizes, or typography
- restyle source cards, image cards, and the report body

This makes it easy to switch between:

- light theme
- softer muted palette
- darker research dashboard style
- more minimal white-paper presentation

## Notes

- Some websites may block scraping or return incomplete HTML.
- Search quality depends on the results returned by DuckDuckGo and Google.
- The quality of the report depends on the API model response and the scraped page content.
- If your system does not expose `python` on the command line, use the Python launcher or full interpreter path.

## Troubleshooting

### No API key found

Make sure one of these is set:

- `NVIDIA_API_KEY`
- `OPENROUTER_API_KEY`
- `API_KEY`

Or place the key in a `.env` file.

### Very few results

Possible reasons:

- the query is too broad or too niche
- target pages blocked scraping
- search engines returned limited links
- the current model response was weak

Try using a more specific query with names, platforms, or keywords.

### Report generated but looks plain

You can directly edit the HTML/CSS in `generate_html_report(...)` to change:

- colors
- spacing
- fonts
- section layout
- source and image card styles

## Summary

Lutervyn Deep Research is a compact research pipeline that combines:

- LLM-assisted search planning
- web scraping
- information extraction
- image discovery
- HTML report generation

It is designed to turn a rough query into a browsable research report that is much easier to inspect than raw terminal output.
