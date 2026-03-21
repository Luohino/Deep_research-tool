import asyncio
import aiohttp
import os
import sys
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import nest_asyncio

nest_asyncio.apply()

import json

# =======================
# Configuration Constants
# =======================
# Using the provided Nvidia API key {saqib tu ye use krle pr fir bhi agr tujhe laga ki khud ka krlu to krliyo jab ye acha response n kre to } 
# OPENROUTER_API_KEY = "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"

# Endpoints
OPENROUTER_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Default LLM model - Llama 3.1 70B Instruct is excellent for research reasoning
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

if not OPENROUTER_API_KEY:
    print("ERROR: API_KEY environment variable is not set.")
    print("Please set it before running this script.")
    sys.exit(1)

# ============================
# Asynchronous Helper Functions
# ============================

async def call_openrouter_async(session, messages, model=DEFAULT_MODEL):
    """
    Asynchronously call the OpenRouter chat completion API with the provided messages.
    Returns the content of the assistant’s reply.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "X-Title": "Custom Deep Researcher",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages
    }
    try:
        async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                result = await resp.json()
                try:
                    return result['choices'][0]['message']['content']
                except (KeyError, IndexError) as e:
                    print("Unexpected OpenRouter response structure:", result)
                    return None
            else:
                text = await resp.text()
                print(f"OpenRouter API error: {resp.status} - {text}")
                return None
    except Exception as e:
        print("Error calling OpenRouter:", e)
        return None

async def generate_search_queries_async(session, user_query):
    """
    Ask the LLM to produce up to four precise search queries (in Python list format)
    based on the user’s query.
    """
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "IMPORTANT: If the query is a person's name or handle (like 'Luohino'), include terms like 'developer', 'github', 'linked-in', and 'projects' to avoid generic results. "
        "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        try:
            # Expect exactly a Python list (e.g., "['query1', 'query2']")
            search_queries = eval(response)
            if isinstance(search_queries, list):
                return search_queries
            else:
                print("LLM did not return a list. Response:", response)
                return []
        except Exception as e:
            print("Error parsing search queries:", e, "\nResponse:", response)
            return []
    return []

from googlesearch import search

async def perform_search_async(session, query):
    """
    Asynchronously perform a search. Tries DuckDuckGo first, then fallbacks to Google.
    """
    links = []
    # Try DuckDuckGo
    try:
        results = DDGS().text(query, max_results=5)
        links = [r['href'] for r in results if 'href' in r]
    except Exception as e:
        print(f"DuckDuckGo search error for '{query}': {e}")
    
    # Fallback to Google if no links found
    if not links:
        try:
            print(f"DuckDuckGo failed, trying Google fallback for '{query}'...")
            results = search(query, num_results=5)
            links = list(results)
        except Exception as e:
            print(f"Google search fallback error: {e}")
            
    return links

async def perform_image_search_async(session, query):
    """
    Asynchronously perform a DuckDuckGo image search.
    """
    images = []
    try:
        results = DDGS().images(query, max_results=3)
        images = [{"url": r['image'], "title": r['title']} for r in results if 'image' in r]
    except Exception as e:
        print(f"Image search error for '{query}': {e}")
    return images

async def fetch_webpage_text_async(session, url):
    """
    Asynchronously retrieve the text content of a webpage using BeautifulSoup
    No API key required!
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=10) as resp:
            if resp.status == 200:
                html = await resp.text()
                soup = BeautifulSoup(html, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                return soup.get_text(separator=' ', strip=True)
            else:
                print(f"Fetch error for {url}: {resp.status}")
                return ""
    except Exception as e:
        print(f"Error fetching webpage text: {e}")
        return ""

async def is_page_useful_async(session, query, text):
    """
    Overriding relevance check to ensure we capture all potential info for the user.
    """
    return True

async def extract_relevant_context_async(session, user_query, search_query, text):
    """
    Extracts relevant context. If LLM fails or is empty, returns a crude snippet.
    """
    prompt = (
        f"You are a research assistant. Extract a concise summary or relevant facts from the text below "
        f"that helps answer: '{user_query}'. If no direct info is found, summarize the page's general purpose. "
    )
    messages = [
        {"role": "system", "content": "You are a concise summarizer."},
        {"role": "user", "content": f"PAGE TEXT:\n{text[:5000]}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response and len(response.strip()) > 10:
        return response.strip()
    return text[:500] + "..." # Fallback to raw snippet

async def get_new_search_queries_async(session, user_query, previous_search_queries, all_contexts):
    """
    Based on the extracted contexts, ask the LLM if more queries are needed.
    """
    context_combined = "\n".join([c['text'] for c in all_contexts])
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with exactly <done>."
        "\nOutput only a Python list or the token <done> without any additional text."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_openrouter_async(session, messages)
    if response:
        cleaned = response.strip()
        if cleaned == "<done>":
            return "<done>"
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
        except Exception:
            return []
    return []

async def generate_final_report_async(session, user_query, contexts):
    """
    Synthesizes the gathered contexts into a final, high-quality report with citations.
    """
    context_str = "\n\n".join([f"SOURCE [{i+1}]: {c['url']}\n{c['text']}" for i, c in enumerate(contexts)])
    
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below, "
        "write a concise, high-impact report that directly addresses the user's query. "
        "Structure the report as follows: "
        "1. **Quick Summary Table**: A markdown table with 'Key Fact' and 'Detail'. Put a blank line BEFORE the table. "
        "2. **The Result**: A structured response. IMPORTANT: Use citations like [1], [2] throughout the text to map to the sources below. "
        "3. **Highlights**: Bullet points of the most important findings with citations. "
        "Do not write long paragraphs. Keep it visual, easy to scan, and deeply cited."
    )
    
    messages = [
        {"role": "system", "content": "You are a world-class researcher. You ALWAYS use citations like [1] in your text."},
        {"role": "user", "content": f"USER QUERY: {user_query}\n\nGATHERED CONTEXTS:\n{context_str}\n\n{prompt}"}
    ]
    
    report_text = await call_openrouter_async(session, messages)
    return report_text

import markdown

def generate_html_report(query, report_text, contexts, images, social_links, all_links, output_path):
    """
    Generates a world-class, Perplexity-style research report with citations and visual gallery.
    """
    # Convert markdown to HTML
    report_html_body = markdown.markdown(report_text, extensions=['tables', 'fenced_code', 'toc'])
    
    # Source Cards (Horizontal Row at Top) - Using ALL links explored for transparency
    sources_html = ""
    for i, link in enumerate(all_links):
        domain = link.split('/')[2].replace('www.', '') if '//' in link else link[:20]
        favicon = f"https://www.google.com/s2/favicons?domain={domain}&sz=64"
        sources_html += f"""
        <a href="{link}" target="_blank" class="source-bubble">
            <img src="{favicon}" class="favicon" alt="icon">
            <div class="source-info">
                <div class="source-num">[{i+1}]</div>
                <div class="source-domain">{domain}</div>
            </div>
        </a>"""
        
    # Images (Grid)
    images_html = ""
    for img in images:
        images_html += f"""
        <div class="image-box">
            <img src="{img['url']}" alt="{img['title']}" loading="lazy">
        </div>"""

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{query} - Deep Research</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #030712;
            --card-bg: #111827;
            --primary: #3b82f6;
            --accent: #60a5fa;
            --text: #f3f4f6;
            --text-muted: #9ca3af;
            --border: rgba(255, 255, 255, 0.08);
        }}
        * {{ box-sizing: border-box; transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1); }}
        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            line-height: 1.6;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .container {{
            max-width: 800px;
            width: 100%;
            padding: 4rem 1.5rem;
        }}
        h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.04em;
            margin-bottom: 2rem;
            background: linear-gradient(to right, #fff, #94a3b8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        /* Sources Scroll Bar */
        .sources-bar {{
            display: flex;
            gap: 0.75rem;
            overflow-x: auto;
            padding-bottom: 1.5rem;
            margin-bottom: 2.5rem;
            scrollbar-width: none;
        }}
        .sources-bar::-webkit-scrollbar {{ display: none; }}
        .source-bubble {{
            flex: 0 0 auto;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            text-decoration: none;
            max-width: 180px;
        }}
        .source-bubble:hover {{ border-color: var(--primary); background: #1f2937; }}
        .favicon {{ width: 20px; height: 20px; border-radius: 4px; }}
        .source-info {{ overflow: hidden; }}
        .source-num {{ font-size: 0.7rem; color: var(--accent); font-weight: 700; }}
        .source-domain {{ font-size: 0.85rem; color: var(--text); white-space: nowrap; text-overflow: ellipsis; overflow: hidden; }}

        /* Image Grid */
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            margin-bottom: 3rem;
        }}
        .image-box {{
            aspect-ratio: 16/10;
            border-radius: 8px;
            overflow: hidden;
            background: var(--card-bg);
        }}
        .image-box img {{ width: 100%; height: 100%; object-fit: cover; }}
        .image-box:hover img {{ transform: scale(1.05); }}

        /* Report Body */
        .report-body {{ font-size: 1.1rem; }}
        .report-body h2 {{ font-size: 1.5rem; margin-top: 3rem; color: white; }}
        .report-body p {{ margin-bottom: 1.5rem; color: #d1d5db; }}
        
        /* Citations */
        sup {{
            color: var(--accent);
            font-weight: 700;
            margin-left: 2px;
            font-size: 0.75rem;
            cursor: pointer;
        }}
        
        /* Table Style */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            font-size: 0.95rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }}
        th {{ background: #1f2937; color: var(--accent); text-align: left; padding: 1rem; text-transform: uppercase; font-size: 0.75rem; }}
        td {{ padding: 1.2rem 1rem; border-top: 1px solid var(--border); }}

        /* Footer */
        .footer {{
            margin-top: 8rem;
            border-top: 1px solid var(--border);
            padding: 4rem 0;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        .fade-in {{ animation: fadeIn 0.8s ease-out forwards; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(20px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    </style>
</head>
<body>
    <div class="container fade-in">
        <header>
            <h1>{query}</h1>
            <div class="sources-bar">
                {sources_html}
            </div>
            {f'<div class="image-grid">{images_html}</div>' if images_html else ''}
        </header>

        <article class="report-body">
            {report_html_body}
        </article>

        <footer class="footer">
            Generated by <b>Lutervyn Deep Research</b> • {import_time()}
        </footer>
    </div>
    
    <script>
        // Simple citation animator or link jump can go here
        document.querySelectorAll('p').forEach(p => {{
            p.innerHTML = p.innerHTML.replace(/\[(\d+)\]/g, '<sup>[$1]</sup>');
        }});
    </script>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    print(f"\n[PERPLEXITY-STYLE REPORT READY] --> {output_path}")

def import_time():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def process_link(session, link, user_query, search_query):
    print(f"Fetching content from: {link}")
    page_text = await fetch_webpage_text_async(session, link)
    if not page_text:
        return None
    usefulness = await is_page_useful_async(session, user_query, page_text)
    print(f"Page usefulness for {link}: {usefulness}")
    if usefulness == "Yes":
        context = await extract_relevant_context_async(session, user_query, search_query, page_text)
        if context:
            return {"url": link, "text": context}
    return None

async def async_main():
    import sys
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        iteration_limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        print(f"\nRunning research for: {user_query} (max iterations: {iteration_limit})")
    else:
        user_query = input("\nEnter your research query/topic: ").strip()
        if not user_query: return
        
        iter_limit_input = input("Enter maximum number of iterations (default 5): ").strip()
        iteration_limit = int(iter_limit_input) if iter_limit_input.isdigit() else 5

    social_links = set()
    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    async with aiohttp.ClientSession() as session:
        new_search_queries = await generate_search_queries_async(session, user_query)
        if not new_search_queries:
            print("No search queries were generated by the LLM. Exiting.")
            return
        all_search_queries.extend(new_search_queries)

        while iteration < iteration_limit:
            print(f"\n=== Iteration {iteration + 1} ===")
            iteration_contexts = []

            # Perform search
            search_tasks = [perform_search_async(session, query) for query in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)

            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    # Detect socials
                    if any(s in link.lower() for s in ["github.com", "x.com", "twitter.com", "linkedin.com", "dev.to"]):
                        social_links.add(link)
                    
                    if link not in unique_links:
                        unique_links[link] = query

            print(f"Aggregated {len(unique_links)} unique links from this iteration.")

            link_tasks = [
                process_link(session, link, user_query, unique_links[link])
                for link in unique_links
            ]
            link_results = await asyncio.gather(*link_tasks)

            for res in link_results:
                if res:
                    iteration_contexts.append(res)

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
            else:
                print("No useful contexts were found in this iteration.")

            new_search_queries = await get_new_search_queries_async(session, user_query, all_search_queries, aggregated_contexts)
            if new_search_queries == "<done>":
                print("LLM indicated that no further research is needed.")
                break
            elif new_search_queries:
                print("LLM provided new search queries:", new_search_queries)
                all_search_queries.extend(new_search_queries)
            else:
                print("LLM did not provide any new search queries. Ending the loop.")
                break

            iteration += 1

        print("\nSearching for relevant images...")
        aggregated_images = []
        # Attempt image search for the main query and some generated queries
        image_queries = [user_query] + (all_search_queries[:2] if all_search_queries else [])
        image_tasks = [perform_image_search_async(session, q) for q in image_queries]
        image_results = await asyncio.gather(*image_tasks)
        for res_list in image_results:
            aggregated_images.extend(res_list)
        # Unique images by URL
        seen_imgs = set()
        final_images = []
        for img in aggregated_images:
            if img['url'] not in seen_imgs:
                seen_imgs.add(img['url'])
                final_images.append(img)
        print(f"Found {len(final_images)} unique images.")

        print("\nGenerating final report...")
        final_report = await generate_final_report_async(session, user_query, aggregated_contexts)
        print("\n==== FINAL REPORT ====\n")
        print(final_report)

        # Create reports directory
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        safe_name = "".join([c if c.isalnum() else "_" for c in user_query[:30]])
        output_file = os.path.join(reports_dir, f"{safe_name}.html")
        
        all_search_links = list(unique_links.keys())
        
        generate_html_report(user_query, final_report, aggregated_contexts, final_images[:6], list(social_links)[:10], all_search_links[:12], output_file)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
