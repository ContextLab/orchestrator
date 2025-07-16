Web Tools
=========

Tools for web scraping, search, and browser automation.

.. automodule:: orchestrator.tools.web_tools
   :members:
   :undoc-members:
   :show-inheritance:

WebSearchTool
-------------

.. autoclass:: orchestrator.tools.web_tools.WebSearchTool
   :members:
   :undoc-members:
   :show-inheritance:

Performs web searches using DuckDuckGo or other search engines.

**Parameters:**

* ``query`` (string, required): Search query
* ``max_results`` (integer, optional): Maximum number of results (default: 10)
* ``region`` (string, optional): Search region (default: "us-en")
* ``safe_search`` (string, optional): Safe search setting ("on", "moderate", "off")

**Returns:**

.. code-block:: python

   {
       "success": True,
       "results": [
           {
               "title": "Page Title",
               "url": "https://example.com",
               "snippet": "Page description...",
               "source": "web"
           }
       ],
       "query": "original query",
       "total_results": 5,
       "search_time": 1.23
   }

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.web_tools import WebSearchTool
   import asyncio
   
   async def search_example():
       tool = WebSearchTool()
       
       results = await tool.execute(
           query="machine learning tutorials",
           max_results=5,
           region="us-en"
       )
       
       if results["success"]:
           for result in results["results"]:
               print(f"Title: {result['title']}")
               print(f"URL: {result['url']}")
               print(f"Snippet: {result['snippet']}")
               print("---")
       
       return results
   
   # Run the search
   asyncio.run(search_example())

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: search_web
       action: search_web
       parameters:
         query: "{{ inputs.search_topic }}"
         max_results: 10
         safe_search: "moderate"

HeadlessBrowserTool
-------------------

.. autoclass:: orchestrator.tools.web_tools.HeadlessBrowserTool
   :members:
   :undoc-members:
   :show-inheritance:

Browser automation tool for content extraction and web scraping.

**Parameters:**

* ``action`` (string, required): Action to perform ("scrape", "screenshot", "interact")
* ``url`` (string, required for scrape/screenshot): Target URL
* ``selector`` (string, optional): CSS selector for specific elements
* ``wait_for`` (string, optional): Element to wait for before proceeding
* ``timeout`` (integer, optional): Timeout in seconds (default: 30)
* ``user_agent`` (string, optional): Custom user agent string

**Actions:**

scrape
~~~~~~

Extract content from a web page:

.. code-block:: python

   result = await browser_tool.execute(
       action="scrape",
       url="https://example.com",
       selector="article",  # Optional: specific content
       wait_for=".content-loaded"  # Optional: wait for element
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "content": "Extracted text content...",
       "html": "<html>...</html>",
       "title": "Page Title",
       "url": "https://example.com",
       "word_count": 542,
       "links": ["https://link1.com", "https://link2.com"],
       "images": ["https://image1.jpg", "https://image2.png"]
   }

screenshot
~~~~~~~~~~

Capture a screenshot of a web page:

.. code-block:: python

   result = await browser_tool.execute(
       action="screenshot",
       url="https://example.com",
       timeout=10
   )

**Returns:**

.. code-block:: python

   {
       "success": True,
       "screenshot_path": "/path/to/screenshot.png",
       "file_size": 102400,
       "dimensions": {"width": 1920, "height": 1080}
   }

interact
~~~~~~~~

Interact with web page elements:

.. code-block:: python

   result = await browser_tool.execute(
       action="interact",
       url="https://example.com",
       interactions=[
           {"type": "click", "selector": "#button"},
           {"type": "type", "selector": "#input", "text": "Hello"},
           {"type": "wait", "selector": ".result"}
       ]
   )

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.web_tools import HeadlessBrowserTool
   import asyncio
   
   async def scrape_example():
       tool = HeadlessBrowserTool()
       
       # Scrape content from a page
       result = await tool.execute(
           action="scrape",
           url="https://news.ycombinator.com",
           selector=".story",
           timeout=15
       )
       
       if result["success"]:
           print(f"Title: {result['title']}")
           print(f"Word count: {result['word_count']}")
           print(f"Content preview: {result['content'][:200]}...")
       
       return result
   
   # Run the scraper
   asyncio.run(scrape_example())

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: scrape_content
       action: scrape_page
       parameters:
         action: "scrape"
         url: "{{ results.search_web.results[0].url }}"
         selector: "article"
         timeout: 30
       dependencies:
         - search_web

Configuration
-------------

Web tools can be configured through the orchestrator configuration:

.. code-block:: yaml

   # config/orchestrator.yaml
   web_tools:
     search:
       default_backend: "duckduckgo"
       max_results: 10
       timeout: 30
       rate_limit: 10  # requests per minute
     
     scraping:
       timeout: 30
       max_content_length: 1048576  # 1MB
       user_agent: "Mozilla/5.0 (compatible; Orchestrator Bot)"
       headers:
         Accept: "text/html,application/xhtml+xml"
       
     browser:
       headless: true
       viewport: {"width": 1920, "height": 1080}
       args:
         - "--no-sandbox"
         - "--disable-dev-shm-usage"

Error Handling
--------------

Web tools handle various error conditions:

Connection Errors
~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "connection_failed",
       "message": "Failed to connect to https://example.com",
       "retry": True,
       "status_code": null
   }

Timeout Errors
~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "timeout",
       "message": "Request timed out after 30 seconds",
       "retry": True
   }

Content Errors
~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "content_error",
       "message": "No content found matching selector",
       "retry": False,
       "status_code": 200
   }

Rate Limiting
~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "rate_limited",
       "message": "Rate limit exceeded, retry after 60 seconds",
       "retry": True,
       "retry_after": 60
   }

Best Practices
--------------

Search Optimization
~~~~~~~~~~~~~~~~~~~

* **Specific Queries**: Use specific, targeted search queries
* **Result Limits**: Set appropriate limits to avoid excessive API usage
* **Result Filtering**: Filter results by relevance or date when possible
* **Error Handling**: Handle rate limits and connection failures gracefully

Web Scraping Ethics
~~~~~~~~~~~~~~~~~~~

* **Respect robots.txt**: Check robots.txt before scraping
* **Rate Limiting**: Don't overwhelm servers with requests
* **User Agent**: Use descriptive, identifiable user agent strings
* **Content Usage**: Respect copyright and terms of service
* **Caching**: Cache scraped content to avoid repeated requests

Performance
~~~~~~~~~~~

* **Async Operations**: Use async/await for concurrent operations
* **Connection Pooling**: Reuse connections when possible
* **Timeout Settings**: Set appropriate timeouts for different operations
* **Resource Cleanup**: Clean up browser instances and temporary files

Security
~~~~~~~~

* **Input Validation**: Validate URLs and parameters
* **Sandboxing**: Run browsers in sandboxed environments
* **Content Filtering**: Filter potentially harmful content
* **Access Control**: Restrict access to internal networks

Examples
--------

Research Assistant
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def research_assistant(topic):
       search_tool = WebSearchTool()
       browser_tool = HeadlessBrowserTool()
       
       # Search for information
       search_results = await search_tool.execute(
           query=topic,
           max_results=5
       )
       
       # Extract content from top result
       if search_results["success"] and search_results["results"]:
           top_url = search_results["results"][0]["url"]
           
           content = await browser_tool.execute(
               action="scrape",
               url=top_url,
               selector="article, .content, main"
           )
           
           return {
               "search_results": search_results,
               "detailed_content": content
           }

Content Monitoring
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def monitor_website(url, selector):
       browser_tool = HeadlessBrowserTool()
       
       # Take screenshot
       screenshot = await browser_tool.execute(
           action="screenshot",
           url=url
       )
       
       # Extract specific content
       content = await browser_tool.execute(
           action="scrape",
           url=url,
           selector=selector
       )
       
       return {
           "screenshot": screenshot,
           "content": content,
           "timestamp": datetime.now().isoformat()
       }

For more examples, see :doc:`../../tutorials/examples/web_scraping`.