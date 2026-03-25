import asyncio
import aiohttp
import os
import sys
import ast
import json
import re
from html import escape
from urllib.parse import urlparse
import markdown
import nest_asyncio
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from datetime import datetime

nest_asyncio.apply()

# =======================
# Configuration Constants
# =======================

# Try loading from .env file first
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENROUTER_API_KEY = os.getenv("NVIDIA_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY", "")

# Endpoints
OPENROUTER_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Default LLM model
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

if not OPENROUTER_API_KEY:
    print("ERROR: No API key found!")
    print("Set NVIDIA_API_KEY, OPENROUTER_API_KEY, or API_KEY environment variable.")
    print("Or create a .env file with: NVIDIA_API_KEY=your_key_here")
    sys.exit(1)

# ============================
# Asynchronous Helper Functions
# ============================

async def call_openrouter_async(session, messages, model=DEFAULT_MODEL):
    """Call the LLM API with the provided messages."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "Lutervyn Deep Research",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "messages": messages}
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    print("Unexpected API response structure:", result)
                    return None
            else:
                text = await resp.text()
                print(f"API error: {resp.status} - {text[:200]}")
                return None
    except Exception as e:
        print("Error calling API:", e)
        return None


async def generate_search_queries_async(session, user_query):
    """Generate diverse search queries for the user's topic."""
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to 5 distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Make queries DIVERSE — include variations for:\n"
        "- Official/primary sources (GitHub, personal sites, LinkedIn)\n"
        "- News articles and blog posts\n"
        "- Forum discussions (Reddit, StackOverflow, HackerNews)\n"
        "- Social media profiles (Twitter/X, dev.to)\n"
        "If the query is about a person, include their name + 'developer', 'github', 'linkedin', 'projects', 'portfolio'.\n"
        "Return ONLY a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant. Return ONLY a Python list."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        search_queries = safe_parse_list_response(response, fallback=[user_query])
        if search_queries:
            return search_queries[:5]
    return [user_query]


from googlesearch import search as google_search


def safe_parse_list_response(response, fallback=None):
    if not response:
        return fallback or []

    cleaned = response.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1].strip()
        if cleaned.startswith("python"):
            cleaned = cleaned[6:].strip()

    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return fallback or []


def truncate_text(text, max_len=80):
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def truncate_url(url, max_len=60):
    clean = (url or "").replace("https://", "").replace("http://", "")
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."

async def perform_search_async(session, query):
    """Perform web search using DuckDuckGo with Google fallback."""
    links = []
    try:
        results = DDGS().text(query, max_results=6)
        links = [r['href'] for r in results if 'href' in r]
    except Exception as e:
        print(f"DuckDuckGo search error for '{query}': {e}")

    if not links:
        try:
            print(f"DuckDuckGo failed, trying Google for '{query}'...")
            results = google_search(query, num_results=6)
            links = list(results)
        except Exception as e:
            print(f"Google search fallback error: {e}")

    return links


async def perform_image_search_async(session, query):
    """Perform DuckDuckGo image search with retry."""
    images = []
    # Add a small delay to avoid rate-limiting
    await asyncio.sleep(1)
    try:
        results = DDGS().images(query, max_results=15)
        images = [
            {
                "url": r["image"],
                "thumbnail": r.get("thumbnail") or r["image"],
                "title": r.get("title", ""),
                "page_url": r.get("url") or r.get("source", ""),
                "domain": urlparse((r.get("url") or r.get("source", ""))).netloc.replace("www.", ""),
            }
            for r in results
            if "image" in r
        ]
    except Exception as e:
        print(f"Image search error for '{query}': {e}")
    return images


async def scrape_page_images_async(session, url):
    """Scrape actual <img> tags from a webpage as image fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    images = []
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                domain = urlparse(url).netloc.replace('www.', '')
                
                for img_tag in soup.find_all("img"):
                    src = img_tag.get("src") or img_tag.get("data-src") or ""
                    if not src:
                        continue
                    # Make absolute URL
                    if src.startswith("//"):
                        src = "https:" + src
                    elif src.startswith("/"):
                        base = urlparse(url)
                        src = f"{base.scheme}://{base.netloc}{src}"
                    elif not src.startswith("http"):
                        continue
                    
                    # Skip tiny images (icons, tracking pixels)
                    width = img_tag.get("width", "")
                    height = img_tag.get("height", "")
                    if width and width.isdigit() and int(width) < 80:
                        continue
                    if height and height.isdigit() and int(height) < 80:
                        continue
                    
                    # Skip common non-content images
                    skip_patterns = ['logo', 'icon', 'badge', 'avatar', 'sprite', 'pixel', 'tracking', '.svg', '1x1', 'spacer']
                    if any(p in src.lower() for p in skip_patterns):
                        continue
                    
                    alt_text = img_tag.get("alt", "") or img_tag.get("title", "")
                    images.append({
                        "url": src,
                        "thumbnail": src,
                        "title": alt_text or f"Image from {domain}",
                        "page_url": url,
                        "domain": domain,
                    })
                    if len(images) >= 10:  # Max 10 per page
                        break
    except Exception as e:
        print(f"  ⚠️ Image scrape error for {url}: {e}")
    return images


async def fetch_webpage_text_async(session, url):
    """Retrieve text content + metadata from a webpage."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract page title
                title = ""
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()

                # Extract meta description & og:image
                meta_desc = ""
                og_image = ""
                for meta in soup.find_all("meta"):
                    if meta.get("name", "").lower() == "description":
                        meta_desc = meta.get("content", "")
                    if meta.get("property", "").lower() == "og:image":
                        og_image = meta.get("content", "")

                # Remove script/style tags
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.extract()

                body_text = soup.get_text(separator=' ', strip=True)

                # Combine metadata + body text
                combined = ""
                if title:
                    combined += f"PAGE TITLE: {title}\n"
                if meta_desc:
                    combined += f"DESCRIPTION: {meta_desc}\n"
                if og_image:
                    combined += f"OG_IMAGE: {og_image}\n"
                combined += f"\nCONTENT:\n{body_text}"

                return combined, title, og_image
            else:
                print(f"Fetch error for {url}: {resp.status}")
                return "", "", ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return "", "", ""


async def extract_relevant_context_async(session, user_query, search_query, text):
    """Extract relevant context from scraped page text."""
    prompt = (
        f"You are a research assistant. Extract ALL relevant facts, details, and information from the text below "
        f"that helps answer: '{user_query}'.\n"
        f"- Include specific names, dates, numbers, URLs, project names, and descriptions.\n"
        f"- If it mentions social profiles, GitHub repos, or portfolio links, INCLUDE those URLs.\n"
        f"- If no direct info is found, give a 1-line summary of what this page is about.\n"
        f"- Be concise but don't skip important details."
    )
    messages = [
        {"role": "system", "content": "You are a precise fact extractor. Keep key details, URLs, and names."},
        {"role": "user", "content": f"PAGE TEXT:\n{text[:8000]}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response and len(response.strip()) > 10:
        return response.strip()
    return text[:600] + "..."


async def get_new_search_queries_async(session, user_query, previous_search_queries, all_contexts):
    """Determine if more research is needed and generate new queries."""
    context_combined = "\n".join([c['text'][:500] for c in all_contexts[-10:]])
    prompt = (
        "You are an analytical research assistant. Based on the original query, searches performed so far, "
        "and the extracted contexts, determine if further research is needed. "
        "If further research is needed, provide up to 4 NEW search queries as a Python list. "
        "If you believe the research is comprehensive enough, respond with exactly <done>.\n"
        "Output ONLY a Python list or the token <done>."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner. Output only a list or <done>."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Queries: {previous_search_queries}\n\nContexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        cleaned = response.strip()
        if "<done>" in cleaned.lower():
            return "<done>"
        return safe_parse_list_response(cleaned, fallback=[])[:4]
    return []


async def generate_final_report_async(session, user_query, contexts):
    """Synthesize gathered contexts into a rich, high-quality report."""
    context_str = "\n\n".join([f"SOURCE [{i+1}]: {c['url']}\n{c['text']}" for i, c in enumerate(contexts)])

    prompt = (
        "You are an elite researcher and writer. Based on the gathered sources below, "
        "write a STUNNING, detailed report that directly addresses the user's query.\n\n"
        "FORMAT YOUR REPORT EXACTLY LIKE THIS:\n\n"
        "## Quick Summary\n"
        "| Key Fact | Detail |\n"
        "|----------|--------|\n"
        "| fact1 | detail1 |\n\n"
        "## Deep Dive\n"
        "Write the main body here. Use **bold** for key terms. "
        "ALWAYS cite sources with [1], [2] etc. throughout the text. "
        "Break into clear subsections with ### headings if the topic is complex.\n\n"
        "## Key Highlights\n"
        "- Bullet points of the most important discoveries with citations [N]\n\n"
        "## Notable Links & Resources\n"
        "- List any important URLs, repos, profiles, or resources discovered\n\n"
        "RULES:\n"
        "- Use vivid language, not boring/generic text\n"
        "- Every claim MUST have a citation like [1] or [2]\n"
        "- Keep section headers clean and professional\n"
        "- If you found specific URLs/profiles in the sources, mention them explicitly\n"
        "- Be thorough but scannable — use bullet points and short paragraphs\n"
        "- Write at least 400 words if there's enough source material"
    )

    messages = [
        {"role": "system", "content": "You are a world-class researcher. You ALWAYS use citations like [1] in your text. Write rich, engaging reports."},
        {"role": "user", "content": f"USER QUERY: {user_query}\n\nGATHERED SOURCES:\n{context_str}\n\n{prompt}"}
    ]

    report_text = await call_openrouter_async(session, messages)
    return report_text if report_text else "No report could be generated. Please check your API key and try again."


async def process_link(session, link, user_query, search_query, source_metadata):
    """Process a single link: fetch, extract context, collect metadata."""
    print(f"  📄 Fetching: {link}")
    page_text, title, og_image = await fetch_webpage_text_async(session, link)
    if not page_text:
        return None

    context = await extract_relevant_context_async(session, user_query, search_query, page_text)
    if context:
        # Store source metadata for rich source cards
        source_metadata[link] = {
            "title": title or (link.split('/')[2] if '//' in link else link[:30]),
            "og_image": og_image,
        }
        return {"url": link, "text": context}
    return None


def generate_html_report(query, report_text, contexts, images, social_links, all_links, source_metadata, output_path):
    """Generate a premium, Perplexity-style research report HTML."""
    # Convert markdown to HTML
    report_html_body = markdown.markdown(
        report_text,
        extensions=['tables', 'fenced_code', 'toc', 'nl2br']
    )

    cited_links = [c["url"] for c in contexts] if contexts else all_links[:20]
    explored_links = [link for link in all_links if link not in cited_links]

    # Source Cards — rich cards with title, domain, and favicon
    sources_html = ""
    for i, link in enumerate(cited_links[:20]):
        domain = link.split('/')[2].replace('www.', '') if '//' in link else link[:20]
        favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        meta = source_metadata.get(link, {})
        title = meta.get("title", domain)
        # Truncate title
        if len(title) > 40:
            title = title[:37] + "..."
        sources_html += f"""
        <a href="{link}" target="_blank" class="source-card" id="source-{i+1}">
            <div class="source-card-top">
                <img src="{favicon}" class="favicon" alt="" onerror="this.style.display='none'">
                <span class="source-num">[{i+1}]</span>
            </div>
            <div class="source-title">{title}</div>
            <div class="source-domain">{domain}</div>
            <div class="source-link-preview">{truncate_url(link, 36)}</div>
        </a>"""

    # Images Grid
    images_html = ""
    for img in images[:6]:
        preview = img.get("thumbnail") or img["url"]
        page_url = img.get("page_url", "")
        image_title = img.get("title", "Image result")
        image_domain = img.get("domain", "")
        images_html += f"""
        <div class="image-box">
            <a href="{img['url']}" target="_blank" class="image-preview-link">
                <img src="{preview}" alt="{image_title}" loading="lazy"
                     onerror="this.parentElement.parentElement.style.display='none'">
            </a>
            <div class="image-meta">
                <div class="image-title">{truncate_text(image_title, 60)}</div>
                <div class="image-links">
                    <a href="{img['url']}" target="_blank">Open image</a>
                    {f'<a href="{page_url}" target="_blank">Source page</a>' if page_url else ''}
                </div>
                {f'<div class="image-domain">{image_domain}</div>' if image_domain else ''}
            </div>
        </div>"""

    # Social Links
    social_html = ""
    social_labels = {
        "github.com": "GitHub", "linkedin.com": "LinkedIn", "twitter.com": "Twitter",
        "x.com": "X", "dev.to": "Dev.to", "medium.com": "Medium",
        "stackoverflow.com": "Stack Overflow", "youtube.com": "YouTube"
    }
    for link in social_links[:8]:
        domain = link.split('/')[2].replace('www.', '') if '//' in link else ""
        label = domain
        for key, text_label in social_labels.items():
            if key in domain:
                label = text_label
                break
        social_html += f'<a href="{link}" target="_blank" class="social-pill">{label}</a>'

    # Source URLs as JSON for JS citation linking
    source_urls_json = json.dumps(cited_links[:20])
    search_trail_html = "".join(
        f'<div class="source-item"><span class="source-item-num">→</span><a href="{link}" target="_blank">{link}</a></div>'
        for link in explored_links[:12]
    )

    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    num_sources = len(cited_links)
    num_contexts = len(contexts)

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{query} — Lutervyn Deep Research</title>
    <meta name="description" content="Deep research report on: {query}">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #f7f4ef;
            --bg-secondary: #efe8de;
            --bg-card: #ffffff;
            --bg-card-hover: #faf7f2;
            --accent-primary: #b7794b;
            --accent-secondary: #8d5b34;
            --accent-glow: rgba(183, 121, 75, 0.12);
            --accent-cyan: #5f7c76;
            --accent-emerald: #7aa38e;
            --text-primary: #1f2933;
            --text-secondary: #52606d;
            --text-muted: #7b8794;
            --border: rgba(82, 96, 109, 0.14);
            --border-hover: rgba(183, 121, 75, 0.28);
            --gradient-1: linear-gradient(135deg, #8d5b34, #b7794b, #d4a373);
            --gradient-2: linear-gradient(135deg, #7aa38e, #5f7c76, #8d5b34);
            --shadow-glow: 0 14px 40px rgba(72, 56, 42, 0.08);
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        html {{ scroll-behavior: smooth; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(180deg, #fbf8f3 0%, #f3ede4 100%);
            color: var(--text-primary);
            line-height: 1.7;
            min-height: 100vh;
            overflow-x: hidden;
        }}

        body::before {{
            content: '';
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background:
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(212, 163, 115, 0.14), transparent),
                radial-gradient(ellipse 60% 40% at 80% 50%, rgba(122, 163, 142, 0.08), transparent),
                radial-gradient(ellipse 60% 40% at 20% 80%, rgba(143, 115, 92, 0.06), transparent);
            pointer-events: none;
            z-index: 0;
        }}

        .container {{
            max-width: 860px;
            width: 100%;
            margin: 0 auto;
            padding: 3rem 1.5rem 6rem;
            position: relative;
            z-index: 1;
        }}

        /* Header */
        .header {{
            margin-bottom: 3rem;
            animation: fadeUp 0.8s ease-out;
        }}

        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--accent-glow);
            border: 1px solid var(--border-hover);
            border-radius: 100px;
            padding: 0.4rem 1rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--accent-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1.5rem;
        }}

        .badge .dot {{
            width: 6px; height: 6px;
            background: var(--accent-emerald);
            border-radius: 50%;
            animation: pulse 2s ease-in-out infinite;
        }}

        h1 {{
            font-size: clamp(2rem, 5vw, 3rem);
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.15;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 1rem;
        }}

        .meta-line {{
            font-size: 0.85rem;
            color: var(--text-muted);
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
        }}
        .meta-line span {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding-right: 1rem;
            border-right: 1px solid var(--border);
        }}
        .meta-line span:last-child {{ border-right: none; padding-right: 0; }}

        /* Stats Bar */
        .stats-bar {{
            display: flex;
            gap: 1rem;
            margin: 2rem 0;
            flex-wrap: wrap;
        }}
        .stat-chip {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 8px 20px rgba(72, 56, 42, 0.04);
        }}
        .stat-chip strong {{ color: var(--text-primary); font-weight: 600; }}

        /* Sources Scroll */
        .section-label {{
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }}

        .sources-scroll {{
            display: flex;
            gap: 0.75rem;
            overflow-x: auto;
            padding-bottom: 0.75rem;
            margin-bottom: 2.5rem;
            scrollbar-width: thin;
            scrollbar-color: var(--bg-card) transparent;
        }}
        .sources-scroll::-webkit-scrollbar {{ height: 4px; }}
        .sources-scroll::-webkit-scrollbar-track {{ background: transparent; }}
        .sources-scroll::-webkit-scrollbar-thumb {{ background: var(--bg-card); border-radius: 4px; }}

        .source-card {{
            flex: 0 0 auto;
            width: 160px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.85rem;
            text-decoration: none;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: all 0.25s ease;
        }}
        .source-card:hover {{
            border-color: var(--accent-primary);
            background: var(--bg-card-hover);
            transform: translateY(-2px);
            box-shadow: 0 24px 60px rgba(66, 47, 30, 0.06);
        }}
        .source-card-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .favicon {{ width: 18px; height: 18px; border-radius: 4px; }}
        .source-num {{
            font-size: 0.65rem;
            font-weight: 700;
            color: var(--accent-secondary);
            background: var(--accent-glow);
            padding: 0.15rem 0.5rem;
            border-radius: 6px;
        }}
        .source-title {{
            font-size: 0.78rem;
            font-weight: 500;
            color: var(--text-primary);
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .source-domain {{
            font-size: 0.68rem;
            color: var(--text-muted);
        }}
        .source-link-preview {{
            font-size: 0.66rem;
            color: var(--accent-cyan);
            opacity: 0.8;
            word-break: break-all;
        }}

        /* Social Pills */
        .social-section {{ margin-bottom: 2.5rem; }}
        .social-pills {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .social-pill {{
            display: inline-flex;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 100px;
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-decoration: none;
            transition: all 0.2s ease;
        }}
        .social-pill:hover {{
            border-color: var(--accent-primary);
            color: var(--text-primary);
            background: var(--bg-card-hover);
        }}

        /* Image Grid */
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-bottom: 3rem;
            border-radius: 12px;
            overflow: hidden;
        }}
        .image-box {{
            aspect-ratio: 16/10;
            overflow: hidden;
            background: var(--bg-card);
            cursor: pointer;
            position: relative;
            display: flex;
            flex-direction: column;
        }}
        .image-preview-link {{
            display: block;
            flex: 1;
        }}
        .image-box img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.4s ease;
        }}
        .image-box:hover img {{ transform: scale(1.08); }}
        .image-meta {{
            padding: 0.75rem;
            border-top: 1px solid var(--border);
            display: grid;
            gap: 0.45rem;
            background: #fcfaf7;
        }}
        .image-title {{
            font-size: 0.78rem;
            line-height: 1.4;
            color: var(--text-primary);
        }}
        .image-links {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .image-links a {{
            color: var(--accent-cyan);
            text-decoration: none;
            font-size: 0.72rem;
        }}
        .image-domain {{
            font-size: 0.68rem;
            color: var(--text-muted);
        }}

        /* Report Body */
        .report-body {{
            font-size: 1.05rem;
            animation: fadeUp 1s ease-out 0.2s both;
        }}

        .report-body h2 {{
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 3rem 0 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}

        .report-body h3 {{
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--accent-secondary);
            margin: 2rem 0 0.75rem;
        }}

        .report-body p {{
            color: var(--text-secondary);
            margin-bottom: 1.25rem;
            line-height: 1.8;
        }}

        .report-body strong {{ color: var(--text-primary); font-weight: 600; }}

        .report-body ul, .report-body ol {{
            padding-left: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .report-body li {{
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
            line-height: 1.7;
        }}
        .report-body li::marker {{ color: var(--accent-primary); }}

        .report-body a {{
            color: var(--accent-cyan);
            text-decoration: none;
            border-bottom: 1px solid transparent;
            transition: border-color 0.2s;
        }}
        .report-body a:hover {{ border-bottom-color: var(--accent-cyan); }}

        /* Citations */
        .citation-link {{
            color: var(--accent-secondary);
            font-weight: 700;
            font-size: 0.72em;
            text-decoration: none;
            background: var(--accent-glow);
            padding: 0.1em 0.4em;
            border-radius: 4px;
            margin: 0 1px;
            transition: all 0.2s;
            vertical-align: super;
        }}
        .citation-link:hover {{
            background: var(--accent-primary);
            color: #fffdf8;
        }}

        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0 2rem;
            font-size: 0.92rem;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        thead {{ background: var(--bg-card); }}
        th {{
            text-align: left;
            padding: 1rem 1.2rem;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent-secondary);
            border-bottom: 1px solid var(--border);
        }}
        td {{
            padding: 1rem 1.2rem;
            border-top: 1px solid var(--border);
            color: var(--text-secondary);
        }}
        tbody tr:hover {{ background: rgba(183, 121, 75, 0.05); }}

        /* Code blocks */
        code {{
            font-family: 'JetBrains Mono', monospace;
            background: var(--bg-card);
            padding: 0.15em 0.4em;
            border-radius: 4px;
            font-size: 0.88em;
            color: var(--accent-cyan);
        }}
        pre {{
            background: #f6f1ea;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
            overflow-x: auto;
            margin: 1.5rem 0;
        }}
        pre code {{ background: none; padding: 0; color: var(--text-secondary); }}

        /* Full Source List */
        .source-list {{
            margin-top: 4rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border);
        }}
        .source-list h2 {{
            font-size: 1.3rem;
            margin-bottom: 1.5rem;
            color: var(--text-primary);
        }}
        .source-item {{
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.75rem 0;
            border-bottom: 1px solid var(--border);
        }}
        .source-item-num {{
            font-size: 0.7rem;
            font-weight: 700;
            color: var(--accent-secondary);
            background: var(--accent-glow);
            padding: 0.2rem 0.5rem;
            border-radius: 6px;
            flex-shrink: 0;
            margin-top: 0.15rem;
        }}
        .source-item a {{
            color: var(--accent-cyan);
            text-decoration: none;
            font-size: 0.88rem;
            word-break: break-all;
            line-height: 1.5;
        }}
        .source-item a:hover {{ text-decoration: underline; }}

        /* Footer */
        .footer {{
            margin-top: 5rem;
            padding: 3rem 0;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            border-top: 1px solid var(--border);
        }}
        .footer-brand {{
            font-weight: 700;
            background: var(--gradient-1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* Animations */
        @keyframes fadeUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.4; }}
        }}

        /* Responsive */
        @media (max-width: 640px) {{
            .container {{ padding: 2rem 1rem 4rem; }}
            .image-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .stats-bar {{ gap: 0.5rem; }}
            .stat-chip {{ padding: 0.5rem 0.8rem; font-size: 0.75rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge"><span class="dot"></span> Deep Research Report</div>
            <h1>{query}</h1>
            <div class="meta-line">
                <span>{timestamp}</span>
                <span>{num_sources} sources explored</span>
                <span>{num_contexts} pages analyzed</span>
            </div>
        </div>

        <div class="stats-bar">
            <div class="stat-chip"><strong>{num_sources}</strong> Sources</div>
            <div class="stat-chip"><strong>{len(images)}</strong> Images</div>
            <div class="stat-chip"><strong>{len(social_links)}</strong> Social Links</div>
            <div class="stat-chip"><strong>{num_contexts}</strong> Analyzed</div>
        </div>

        <div class="section-label">Sources</div>
        <div class="sources-scroll">
            {sources_html}
        </div>

        {f'<div class="section-label">Images</div><div class="image-grid">{images_html}</div>' if images_html else ''}

        {f'<div class="social-section"><div class="section-label">Social & Profiles</div><div class="social-pills">{social_html}</div></div>' if social_html else ''}

        <article class="report-body">
            {report_html_body}
        </article>

        <div class="source-list">
            <h2>Cited Sources</h2>
            {"".join(f'<div class="source-item"><span class="source-item-num">[{i+1}]</span><a href="{link}" target="_blank">{link}</a></div>' for i, link in enumerate(cited_links[:20]))}
        </div>

        {f'<div class="source-list"><h2>Search Trail</h2>{search_trail_html}</div>' if explored_links else ''}

        <footer class="footer">
            Powered by <span class="footer-brand">Lutervyn Deep Research</span><br>
            <span style="font-size: 0.7rem; margin-top: 0.5rem; display: block;">
                {timestamp} • {num_sources} sources • {num_contexts} pages analyzed
            </span>
        </footer>
    </div>

    <script>
        // Make citations clickable
        const sourceUrls = """ + source_urls_json + """;

        function linkCitations(container) {
            const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
            const textNodes = [];
            while (walker.nextNode()) textNodes.push(walker.currentNode);

            textNodes.forEach(function(node) {
                const text = node.textContent;
                if (/\[\d+\]/.test(text)) {
                    const span = document.createElement('span');
                    span.innerHTML = text.replace(/\[(\d+)\]/g, function(match, num) {
                        const idx = parseInt(num) - 1;
                        if (idx >= 0 && idx < sourceUrls.length) {
                            return '<a href="' + sourceUrls[idx] + '" target="_blank" class="citation-link" title="' + sourceUrls[idx] + '">[' + num + ']</a>';
                        }
                        return '<span class="citation-link">[' + num + ']</span>';
                    });
                    node.parentNode.replaceChild(span, node);
                }
            });
        }

        linkCitations(document.querySelector('.report-body'));

        // Stagger animations
        document.querySelectorAll('.source-card').forEach(function(card, i) {
            card.style.animation = 'fadeUp 0.5s ease-out ' + (i * 0.05) + 's both';
        });
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\n✅ Report saved → {output_path}")


async def process_link_wrapper(session, link, user_query, search_query, source_metadata):
    """Wrapper for process_link with error handling."""
    try:
        return await process_link(session, link, user_query, search_query, source_metadata)
    except Exception as e:
        print(f"  ❌ Error processing {link}: {e}")
        return None


async def async_main():
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        iteration_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"\n🔬 Running deep research for: {user_query} (max iterations: {iteration_limit})")
    else:
        user_query = input("\n🔍 Enter your research query/topic: ").strip()
        if not user_query:
            return
        iter_limit_input = input("Enter max iterations (default 5): ").strip()
        iteration_limit = int(iter_limit_input) if iter_limit_input.isdigit() else 5

    social_links = set()
    aggregated_contexts = []
    all_search_queries = []
    all_explored_links = []  # Accumulate ALL links across ALL iterations
    all_explored_set = set()  # For dedup
    source_metadata = {}  # Store title/og:image per URL
    iteration = 0

    async with aiohttp.ClientSession() as session:
        print("\n🧠 Generating search queries...")
        new_search_queries = await generate_search_queries_async(session, user_query)
        if not new_search_queries:
            print("❌ No search queries generated. Exiting.")
            return
        print(f"   Generated: {new_search_queries}")
        all_search_queries.extend(new_search_queries)

        while iteration < iteration_limit:
            print(f"\n{'='*50}")
            print(f"📡 Iteration {iteration + 1}/{iteration_limit}")
            print(f"{'='*50}")

            # Perform searches
            search_tasks = [perform_search_async(session, q) for q in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)

            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    # Detect social profiles
                    social_domains = ["github.com", "x.com", "twitter.com", "linkedin.com",
                                      "dev.to", "medium.com", "stackoverflow.com", "youtube.com"]
                    if any(s in link.lower() for s in social_domains):
                        social_links.add(link)

                    if link not in all_explored_set:
                        unique_links[link] = query
                        all_explored_set.add(link)
                        all_explored_links.append(link)

            print(f"   🔗 Found {len(unique_links)} new unique links")

            # Process all links
            link_tasks = [
                process_link_wrapper(session, link, user_query, unique_links[link], source_metadata)
                for link in unique_links
            ]
            link_results = await asyncio.gather(*link_tasks)

            new_contexts = [r for r in link_results if r]
            aggregated_contexts.extend(new_contexts)
            print(f"   ✅ Extracted {len(new_contexts)} contexts (total: {len(aggregated_contexts)})")

            if not new_contexts:
                print("   ⚠️ No useful contexts found this iteration.")

            # Check if more research needed
            new_search_queries = await get_new_search_queries_async(
                session, user_query, all_search_queries, aggregated_contexts
            )
            if new_search_queries == "<done>":
                print("\n🏁 LLM says research is comprehensive enough.")
                break
            elif new_search_queries:
                print(f"   🔄 New queries: {new_search_queries}")
                all_search_queries.extend(new_search_queries)
            else:
                print("   🏁 No more queries. Ending research.")
                break

            iteration += 1

        # Image search — try DDG first, then fall back to og:images and page scraping
        print("\n🖼️ Searching for images...")
        aggregated_images = []
        
        # Method 1: DuckDuckGo image search (run queries sequentially to avoid rate-limit)
        image_queries = [user_query] + (all_search_queries[:2] if all_search_queries else [])
        for iq in image_queries:
            result = await perform_image_search_async(session, iq)
            aggregated_images.extend(result)
            if len(aggregated_images) >= 30:
                break
        
        print(f"   DDG images: {len(aggregated_images)}")
        
        # Method 2: Fallback — collect og:images from scraped pages
        if len(aggregated_images) < 4:
            print("   Trying og:image fallback...")
            for link, meta in source_metadata.items():
                og_img = meta.get("og_image", "")
                if og_img and og_img.startswith("http"):
                    domain = urlparse(link).netloc.replace('www.', '')
                    aggregated_images.append({
                        "url": og_img,
                        "thumbnail": og_img,
                        "title": meta.get("title", f"Image from {domain}"),
                        "page_url": link,
                        "domain": domain,
                    })
            print(f"   After og:image fallback: {len(aggregated_images)}")
        
        # Method 3: Fallback — scrape actual <img> tags from top pages
        if len(aggregated_images) < 4:
            print("   Scraping images from top pages...")
            scrape_urls = all_explored_links[:5]
            for surl in scrape_urls:
                scraped = await scrape_page_images_async(session, surl)
                aggregated_images.extend(scraped)
                if len(aggregated_images) >= 30:
                    break
            print(f"   After page scrape: {len(aggregated_images)}")

        # Deduplicate images
        seen_imgs = set()
        final_images = []
        for img in aggregated_images:
            if img['url'] not in seen_imgs:
                seen_imgs.add(img['url'])
                final_images.append(img)
        print(f"   ✅ Total unique images: {len(final_images)}")

        # Generate report
        print("\n📝 Generating final report...")
        if aggregated_contexts:
            final_report = await generate_final_report_async(session, user_query, aggregated_contexts)
        else:
            final_report = (
                f"## ⚠️ Limited Results\n\n"
                f"The research for **\"{user_query}\"** did not yield enough web content to generate a comprehensive report.\n\n"
                f"### Possible reasons:\n"
                f"- The topic may be too niche or private\n"
                f"- Web scraping may have been blocked by some sites\n"
                f"- Search engines returned limited results\n\n"
                f"### Links explored:\n" +
                "\n".join([f"- [{link}]({link})" for link in all_explored_links[:10]])
            )

        print("\n" + "=" * 50)
        print("📋 FINAL REPORT")
        print("=" * 50)
        print(final_report)

        # Save HTML report
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        safe_name = "".join([c if c.isalnum() else "_" for c in user_query[:40]])
        output_file = os.path.join(reports_dir, f"{safe_name}.html")

        generate_html_report(
            query=user_query,
            report_text=final_report,
            contexts=aggregated_contexts,
            images=final_images,
            social_links=list(social_links)[:10],
            all_links=all_explored_links[:20],
            source_metadata=source_metadata,
            output_path=output_file
        )

        print(f"\n🎉 Done! Open the report: {output_file}")


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
