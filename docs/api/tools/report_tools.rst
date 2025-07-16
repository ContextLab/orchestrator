Report Tools
============

Tools for document generation, report creation, and PDF compilation.

.. automodule:: orchestrator.tools.report_tools
   :members:
   :undoc-members:
   :show-inheritance:

ReportGeneratorTool
-------------------

.. autoclass:: orchestrator.tools.report_tools.ReportGeneratorTool
   :members:
   :undoc-members:
   :show-inheritance:

Generates comprehensive reports from various data sources in markdown format.

**Parameters:**

* ``title`` (string, required): Report title
* ``template`` (string, optional): Report template ("research", "business", "technical", "custom")
* ``data`` (object, optional): Data to include in report
* ``search_results`` (object, optional): Web search results
* ``extraction_results`` (object, optional): Content extraction results
* ``findings`` (array, optional): Key findings to highlight
* ``recommendations`` (array, optional): Recommendations to include
* ``sections`` (array, optional): Custom sections to include
* ``metadata`` (object, optional): Additional metadata

**Templates:**

research
~~~~~~~~

Academic/research-style report:

.. code-block:: python

   result = await report_tool.execute(
       title="AI Research Report",
       template="research",
       search_results=search_data,
       findings=[
           "AI adoption increased 40% in 2024",
           "LLMs show improved reasoning capabilities",
           "Ethical AI frameworks gaining adoption"
       ],
       recommendations=[
           "Invest in AI training programs",
           "Develop ethical AI guidelines",
           "Monitor regulatory developments"
       ]
   )

business
~~~~~~~~

Business-focused report with executive summary:

.. code-block:: python

   result = await report_tool.execute(
       title="Market Analysis Report",
       template="business",
       data=market_data,
       sections=["executive_summary", "market_trends", "competitive_analysis", "recommendations"]
   )

technical
~~~~~~~~~

Technical documentation style:

.. code-block:: python

   result = await report_tool.execute(
       title="System Architecture Report",
       template="technical", 
       data=system_data,
       sections=["overview", "architecture", "implementation", "performance", "security"]
   )

custom
~~~~~~

Custom template with user-defined structure:

.. code-block:: python

   result = await report_tool.execute(
       title="Custom Report",
       template="custom",
       sections=[
           {"name": "Introduction", "content": "..."},
           {"name": "Analysis", "content": "..."},
           {"name": "Conclusions", "content": "..."}
       ]
   )

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.report_tools import ReportGeneratorTool
   import asyncio
   
   async def generate_research_report():
       report_tool = ReportGeneratorTool()
       
       # Sample data
       search_results = {
           "query": "renewable energy trends 2024",
           "results": [
               {
                   "title": "Solar Energy Breakthroughs",
                   "url": "https://example.com/solar",
                   "snippet": "Recent advances in solar technology..."
               },
               {
                   "title": "Wind Power Growth",
                   "url": "https://example.com/wind", 
                   "snippet": "Wind energy capacity increased..."
               }
           ]
       }
       
       # Generate report
       result = await report_tool.execute(
           title="Renewable Energy Trends 2024",
           template="research",
           search_results=search_results,
           findings=[
               "Solar efficiency improved by 15%",
               "Wind capacity grew 20% globally",
               "Storage technology advancing rapidly"
           ],
           recommendations=[
               "Increase renewable energy investment",
               "Develop grid storage solutions", 
               "Support policy initiatives"
           ],
           metadata={
               "author": "AI Research Assistant",
               "date": "2024-01-15",
               "version": "1.0"
           }
       )
       
       return result
   
   # Generate report
   asyncio.run(generate_research_report())

**Returns:**

.. code-block:: python

   {
       "success": True,
       "markdown": "# Renewable Energy Trends 2024\n\n...",
       "word_count": 1250,
       "sections": 6,
       "template": "research",
       "metadata": {
           "generated_at": "2024-01-15T10:30:00Z",
           "generator": "ReportGeneratorTool v1.0"
       }
   }

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: generate_report
       action: generate_report
       parameters:
         title: "{{ inputs.report_title }}"
         template: "research"
         search_results: "{{ results.web_search }}"
         findings: "{{ results.analysis.key_points }}"
         recommendations: "{{ results.analysis.recommendations }}"
       dependencies:
         - web_search
         - analysis

PDFCompilerTool
---------------

.. autoclass:: orchestrator.tools.report_tools.PDFCompilerTool
   :members:
   :undoc-members:
   :show-inheritance:

Compiles markdown content to PDF using pandoc with automatic installation and cross-platform support.

**Parameters:**

* ``markdown_content`` (string, required): Markdown content to compile
* ``output_path`` (string, required): Output PDF file path  
* ``title`` (string, optional): Document title
* ``author`` (string, optional): Document author
* ``template`` (string, optional): Pandoc template ("default", "eisvogel", "elegant")
* ``options`` (object, optional): Additional pandoc options
* ``toc`` (boolean, optional): Include table of contents (default: True)
* ``page_size`` (string, optional): Page size ("a4", "letter", "legal")
* ``margin`` (string, optional): Page margins ("1in", "2cm")

**Example Usage:**

.. code-block:: python

   from orchestrator.tools.report_tools import PDFCompilerTool
   import asyncio
   
   async def compile_to_pdf():
       pdf_tool = PDFCompilerTool()
       
       markdown_content = """
   # Research Report
   
   ## Executive Summary
   
   This report analyzes current trends in renewable energy...
   
   ## Key Findings
   
   1. Solar efficiency improvements
   2. Wind capacity growth
   3. Storage technology advances
   
   ## Recommendations
   
   - Increase investment in solar R&D
   - Expand wind farm development
   - Support battery technology research
   """
       
       result = await pdf_tool.execute(
           markdown_content=markdown_content,
           output_path="./reports/renewable_energy_report.pdf",
           title="Renewable Energy Trends 2024",
           author="AI Research Assistant",
           template="eisvogel",
           toc=True,
           page_size="a4",
           margin="1in"
       )
       
       return result
   
   # Compile PDF
   asyncio.run(compile_to_pdf())

**Returns:**

.. code-block:: python

   {
       "success": True,
       "pdf_path": "./reports/renewable_energy_report.pdf",
       "file_size": 245760,
       "pages": 8,
       "compilation_time": 2.34,
       "pandoc_version": "3.1.2",
       "options_used": [
           "--toc",
           "--template=eisvogel",
           "--pdf-engine=xelatex"
       ]
   }

**Pipeline Usage:**

.. code-block:: yaml

   steps:
     - id: compile_pdf
       action: compile_pdf
       parameters:
         markdown_content: "{{ results.generate_report.markdown }}"
         output_path: "./output/{{ inputs.filename }}.pdf"
         title: "{{ inputs.report_title }}"
         author: "Orchestrator Framework"
         template: "eisvogel"
         toc: true
       dependencies:
         - generate_report

Pandoc Installation
-------------------

The PDFCompilerTool automatically installs pandoc if not available:

**Windows:**

.. code-block:: python

   # Automatic installation via chocolatey or direct download
   await pdf_tool.install_pandoc()  # Handles Windows installation

**macOS:**

.. code-block:: python

   # Automatic installation via homebrew or direct download
   await pdf_tool.install_pandoc()  # Handles macOS installation

**Linux:**

.. code-block:: python

   # Automatic installation via package manager
   await pdf_tool.install_pandoc()  # Handles Linux installation

Manual Installation Check:

.. code-block:: python

   # Check if pandoc is available
   is_available = await pdf_tool.check_pandoc_availability()
   
   if not is_available:
       # Install pandoc
       install_result = await pdf_tool.install_pandoc()
       
       if install_result["success"]:
           print(f"Pandoc installed: {install_result['version']}")

Templates
---------

Default Template
~~~~~~~~~~~~~~~~

Basic pandoc template with standard formatting:

.. code-block:: python

   result = await pdf_tool.execute(
       markdown_content=content,
       output_path="report.pdf",
       template="default"
   )

Eisvogel Template
~~~~~~~~~~~~~~~~~

Professional template with enhanced styling:

.. code-block:: python

   result = await pdf_tool.execute(
       markdown_content=content,
       output_path="report.pdf", 
       template="eisvogel",
       options={
           "titlepage": True,
           "logo": "./assets/logo.png",
           "colorlinks": True
       }
   )

Custom Template
~~~~~~~~~~~~~~~

Use custom pandoc templates:

.. code-block:: python

   result = await pdf_tool.execute(
       markdown_content=content,
       output_path="report.pdf",
       template="./templates/custom.latex",
       options={
           "variable": {
               "company": "My Company",
               "department": "Research Division"
           }
       }
   )

Advanced Options
----------------

Font Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   result = await pdf_tool.execute(
       markdown_content=content,
       output_path="report.pdf",
       options={
           "mainfont": "Arial",
           "sansfont": "Helvetica",
           "monofont": "Courier New",
           "fontsize": "12pt"
       }
   )

Image Handling
~~~~~~~~~~~~~~

.. code-block:: python

   result = await pdf_tool.execute(
       markdown_content=content,
       output_path="report.pdf",
       options={
           "dpi": 300,
           "extract-media": "./images/",
           "resource-path": ["./assets/", "./images/"]
       }
   )

Page Layout
~~~~~~~~~~~

.. code-block:: python

   result = await pdf_tool.execute(
       markdown_content=content,
       output_path="report.pdf",
       page_size="a4",
       margin="2cm",
       options={
           "geometry": "margin=2cm",
           "header-includes": [
               "\\usepackage{fancyhdr}",
               "\\pagestyle{fancy}"
           ]
       }
   )

Error Handling
--------------

Pandoc Not Found
~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "pandoc_not_found",
       "message": "Pandoc is not installed or not found in PATH",
       "installation_attempted": True,
       "install_error": "Permission denied"
   }

Compilation Errors
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "compilation_failed",
       "message": "Pandoc compilation failed",
       "stderr": "! LaTeX Error: File 'eisvogel.sty' not found",
       "suggestion": "Install required LaTeX packages"
   }

File System Errors
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   {
       "success": False,
       "error": "file_system_error",
       "message": "Cannot write to output path",
       "output_path": "/protected/path/report.pdf",
       "suggestion": "Check file permissions"
   }

Best Practices
--------------

Report Generation
~~~~~~~~~~~~~~~~~

* **Clear Structure**: Use consistent section organization
* **Rich Content**: Include relevant data, charts, and findings
* **Professional Format**: Use appropriate templates and styling
* **Metadata**: Include author, date, and version information
* **Validation**: Validate markdown syntax before compilation

.. code-block:: python

   async def generate_professional_report(data):
       report_tool = ReportGeneratorTool()
       pdf_tool = PDFCompilerTool()
       
       # Generate structured report
       report = await report_tool.execute(
           title=data["title"],
           template="research",
           search_results=data["search_results"],
           findings=data["findings"],
           recommendations=data["recommendations"],
           metadata={
               "author": "Research Team",
               "date": datetime.now().isoformat(),
               "version": "1.0",
               "confidentiality": "Internal Use"
           }
       )
       
       # Compile to PDF with professional styling
       pdf = await pdf_tool.execute(
           markdown_content=report["markdown"],
           output_path=f"./reports/{data['filename']}.pdf",
           title=data["title"],
           author="Research Team",
           template="eisvogel",
           toc=True,
           options={
               "titlepage": True,
               "colorlinks": True,
               "logo": "./assets/company_logo.png"
           }
       )
       
       return {
           "report": report,
           "pdf": pdf
       }

PDF Compilation
~~~~~~~~~~~~~~~

* **Template Selection**: Choose appropriate templates for content type
* **Resource Management**: Ensure images and assets are accessible
* **Error Handling**: Handle pandoc installation and compilation errors
* **File Organization**: Use organized output directory structure
* **Quality Control**: Verify PDF output quality and completeness

Performance
~~~~~~~~~~~

* **Batch Processing**: Compile multiple reports efficiently
* **Caching**: Cache template and resource files
* **Parallel Processing**: Generate multiple reports concurrently
* **Resource Cleanup**: Clean up temporary files

.. code-block:: python

   async def batch_report_generation(report_configs):
       report_tool = ReportGeneratorTool()
       pdf_tool = PDFCompilerTool()
       
       # Generate reports in parallel
       report_tasks = [
           report_tool.execute(**config)
           for config in report_configs
       ]
       
       reports = await asyncio.gather(*report_tasks)
       
       # Compile PDFs in parallel
       pdf_tasks = [
           pdf_tool.execute(
               markdown_content=report["markdown"],
               output_path=f"./output/report_{i}.pdf",
               template="eisvogel"
           )
           for i, report in enumerate(reports)
       ]
       
       pdfs = await asyncio.gather(*pdf_tasks)
       
       return list(zip(reports, pdfs))

Configuration
-------------

Report tools can be configured globally:

.. code-block:: yaml

   # config/orchestrator.yaml
   report_tools:
     generator:
       default_template: "research"
       include_metadata: true
       author: "AI Assistant"
       logo_path: "./assets/logo.png"
     
     pdf_compiler:
       pandoc_path: "/usr/local/bin/pandoc"
       default_template: "eisvogel"
       default_options:
         toc: true
         number_sections: true
         colorlinks: true
       output_directory: "./reports"
       temp_directory: "./tmp"

Examples
--------

Complete Research Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def research_to_pdf_pipeline(topic):
       from orchestrator.tools import WebSearchTool, ReportGeneratorTool, PDFCompilerTool
       
       # Search for information
       search_tool = WebSearchTool()
       search_results = await search_tool.execute(
           query=topic,
           max_results=10
       )
       
       # Generate report
       report_tool = ReportGeneratorTool()
       report = await report_tool.execute(
           title=f"Research Report: {topic}",
           template="research",
           search_results=search_results,
           findings=[
               "Key finding 1 based on search results",
               "Key finding 2 based on analysis",
               "Key finding 3 based on synthesis"
           ],
           recommendations=[
               "Recommendation 1",
               "Recommendation 2",
               "Recommendation 3"
           ]
       )
       
       # Compile to PDF
       pdf_tool = PDFCompilerTool()
       pdf = await pdf_tool.execute(
           markdown_content=report["markdown"],
           output_path=f"./reports/{topic.replace(' ', '_')}.pdf",
           title=f"Research Report: {topic}",
           author="AI Research Assistant",
           template="eisvogel",
           toc=True
       )
       
       return {
           "search": search_results,
           "report": report,
           "pdf": pdf
       }

Business Report Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   async def business_report_pipeline(data):
       report_tool = ReportGeneratorTool()
       pdf_tool = PDFCompilerTool()
       
       # Generate business report
       report = await report_tool.execute(
           title="Quarterly Business Review",
           template="business",
           data=data,
           sections=[
               "executive_summary",
               "financial_performance", 
               "market_analysis",
               "operational_metrics",
               "strategic_initiatives",
               "recommendations"
           ],
           metadata={
               "quarter": "Q4 2024",
               "department": "Strategy",
               "confidentiality": "Confidential"
           }
       )
       
       # Compile with business template
       pdf = await pdf_tool.execute(
           markdown_content=report["markdown"],
           output_path="./reports/q4_2024_business_review.pdf",
           title="Q4 2024 Business Review",
           author="Strategy Team",
           template="eisvogel",
           options={
               "titlepage": True,
               "logo": "./assets/company_logo.png",
               "toc-depth": 3,
               "colorlinks": True
           }
       )
       
       return {"report": report, "pdf": pdf}

For more examples, see :doc:`../../tutorials/examples/report_generation`.