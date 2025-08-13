from serpapi import GoogleSearch
from ddgs import DDGS
import os

class WebSearchAgent:
    def __init__(self, use_serpapi=False):
        self.use_serpapi = use_serpapi
        self.serpapi_key = os.environ.get("SERPAPI_API_KEY")  # Set your key in env
        if use_serpapi and not self.serpapi_key:
            raise ValueError("SERPAPI_API_KEY is not set in the environment.")

    def search(self, query: str, num_results: int = 5):
        if self.use_serpapi:
            return self._search_serpapi(query, num_results)
        else:
            return self._search_ddg(query, num_results)

    def _search_serpapi(self, query, num_results):
        params = {
            "q": query,
            "api_key": self.serpapi_key,
            "num": num_results,
            "engine": "google",
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        return self._extract_serpapi_results(results, num_results)

    def _extract_serpapi_results(self, results, num_results):
        snippets = []
        for r in results.get("organic_results", [])[:num_results]:
            snippets.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet")
            })
        return snippets

    def _search_ddg(self, query, num_results):
        with DDGS() as ddgs:
            return list(ddgs.text(query, region='us-en', safesearch='Off', max_results=num_results))

    def format_results(self, results):
        formatted = ""
        for i, r in enumerate(results):
            title = r.get("title", "")
            body = r.get("body", "")     # DuckDuckGo snippet
            link = r.get("href", "")     # DuckDuckGo link
            formatted += f"{i+1}. {title}\n{body}\n{link}\n\n"
        return formatted.strip()

# Example usage
if __name__ == "__main__":
    agent = WebSearchAgent(use_serpapi=False)
    query = "latest research on pancreatic cyst classification using AI"
    results = agent.search(query)
    print(agent.format_results(results))

