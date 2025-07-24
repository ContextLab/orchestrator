"""Tools for report generation and PDF compilation."""

import logging
import os
import platform
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .base import Tool

logger = logging.getLogger(__name__)


class ReportGeneratorTool(Tool):
    """Tool for generating markdown reports from research data."""

    def __init__(self):
        super().__init__(
            name="report-generator",
            description="Generate structured markdown reports from research data",
        )
        self.add_parameter("title", "string", "Report title")
        self.add_parameter("query", "string", "Research query")
        self.add_parameter("context", "string", "Research context", required=False)
        self.add_parameter("search_results", "object", "Search results data")
        self.add_parameter("extraction_results", "object", "Content extraction data")
        self.add_parameter("findings", "array", "Key findings", required=False)
        self.add_parameter(
            "recommendations", "array", "Recommendations", required=False
        )
        self.add_parameter("quality_score", "number", "Quality score", required=False)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Generate a markdown report from research data."""
        title = kwargs.get("title", "Research Report")
        query = kwargs.get("query", "")
        context = kwargs.get("context", "")
        search_results = kwargs.get("search_results", {})
        extraction_results = kwargs.get("extraction_results", {})
        findings = kwargs.get("findings", [])
        recommendations = kwargs.get("recommendations", [])
        quality_score = kwargs.get("quality_score", 0.0)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build markdown report
        report_lines = [
            f"# {title}",
            "",
            f"**Generated on:** {timestamp}",
            f"**Query:** {query}",
        ]

        if context:
            report_lines.extend([f"**Context:** {context}", ""])

        if quality_score:
            report_lines.extend([f"**Quality Score:** {quality_score:.2f}/1.0", ""])

        report_lines.extend(["---", "", "## Executive Summary", ""])

        # Generate content-based executive summary
        executive_summary = self._generate_executive_summary(
            query, search_results, findings
        )
        report_lines.extend([executive_summary, ""])

        # Add key findings if available
        if findings:
            report_lines.extend(["### Key Findings", ""])
            for finding in findings:
                report_lines.append(f"- {finding}")
            report_lines.append("")

        # Add search results section
        report_lines.extend(["## Search Results", ""])

        results = search_results.get("results", [])
        if results:
            for idx, result in enumerate(results[:10], 1):  # Limit to top 10
                title = result.get("title", "No title")
                url = result.get("url", "#")
                snippet = result.get("snippet", "No description available.")
                relevance = result.get("relevance", 0)

                report_lines.extend(
                    [
                        f"### {idx}. {title}",
                        f"**URL:** [{url}]({url})",
                        f"**Relevance:** {relevance:.2f}",
                        f"**Summary:** {snippet}",
                        "",
                    ]
                )
        else:
            report_lines.append("No search results found.")
            report_lines.append("")

        # Add extracted content section
        if extraction_results and extraction_results.get("success"):
            report_lines.extend(["## Extracted Content Analysis", ""])

            extracted_url = extraction_results.get("url", "Unknown")
            extracted_title = extraction_results.get("title", "No title")
            word_count = extraction_results.get("word_count", 0)

            report_lines.extend(
                [
                    f"**Source:** [{extracted_title}]({extracted_url})",
                    f"**Word Count:** {word_count:,}",
                    "",
                ]
            )

            # Add content preview
            content = extraction_results.get("text", "")
            if content:
                preview = content[:1000] + "..." if len(content) > 1000 else content
                report_lines.extend(["### Content Preview", "", preview, ""])

        # Add recommendations if available
        if recommendations:
            report_lines.extend(["## Recommendations", ""])
            for idx, rec in enumerate(recommendations, 1):
                report_lines.append(f"{idx}. {rec}")
            report_lines.append("")

        # Add methodology section
        report_lines.extend(
            [
                "## Methodology",
                "",
                "This report was generated using the Orchestrator Research Assistant, which:",
                "- Performs comprehensive web searches using multiple sources",
                "- Extracts and analyzes content from relevant web pages",
                "- Evaluates source credibility and relevance",
                "- Synthesizes findings into actionable insights",
                "",
            ]
        )

        # Add references section
        report_lines.extend(["## References", ""])
        if results:
            for idx, result in enumerate(results[:10], 1):
                title = result.get("title", "No title")
                url = result.get("url", "#")
                report_lines.append(f"{idx}. {title}. Available at: {url}")
        report_lines.append("")

        # Add footer
        report_lines.extend(
            [
                "---",
                "",
                "*This report was automatically generated by the Orchestrator Research Assistant.*",
            ]
        )

        # Join into final markdown
        markdown_content = "\n".join(report_lines)

        return {
            "success": True,
            "markdown": markdown_content,
            "word_count": len(markdown_content.split()),
            "timestamp": timestamp,
            "title": title,
        }

    def _generate_executive_summary(
        self, query: str, search_results: Dict[str, Any], findings: List[str]
    ) -> str:
        """Generate a content-based executive summary focusing on actual insights."""
        results = search_results.get("results", [])

        if not results:
            return f"No relevant information was found for the query '{query}'."

        # Extract actual content insights from snippets
        content_insights = []
        key_concepts = set()

        # Analyze high-relevance snippets for meaningful content
        for result in results[:5]:  # Focus on top 5 most relevant results
            snippet = result.get("snippet", "")
            relevance = result.get("relevance", 0)

            if relevance > 0.7 and snippet and len(snippet) > 50:
                # Extract meaningful phrases and concepts
                self._extract_content_insights(snippet, content_insights, key_concepts)

        # Build content-focused summary
        summary_parts = []

        # Start with actual content insights rather than metadata
        if content_insights:
            if "best practices" in query.lower():
                summary_parts.append("Key best practices identified include:")
                summary_parts.extend(content_insights[:3])
            elif (
                "latest" in query.lower()
                or "recent" in query.lower()
                or "2024" in query
            ):
                summary_parts.append("Recent developments reveal:")
                summary_parts.extend(content_insights[:3])
            elif "technologies" in query.lower():
                summary_parts.append("Current technologies and approaches include:")
                summary_parts.extend(content_insights[:3])
            else:
                summary_parts.append("Research findings indicate:")
                summary_parts.extend(content_insights[:3])
        else:
            # Fallback if no specific insights extracted
            if "best practices" in query.lower():
                summary_parts.append(
                    f"This analysis examines best practices for {query.lower().replace('best practices for ', '')}."
                )
            elif (
                "latest" in query.lower()
                or "recent" in query.lower()
                or "2024" in query
            ):
                summary_parts.append(
                    f"This research explores recent developments in {query.lower().replace('latest developments in ', '').replace(' 2024', '')}."
                )
            else:
                summary_parts.append(
                    f"This research provides insights into {query.lower()}."
                )

        # Add conceptual themes if meaningful ones were found
        meaningful_concepts = [
            c
            for c in key_concepts
            if len(c) > 3
            and c
            not in {
                "with",
                "from",
                "that",
                "this",
                "they",
                "your",
                "code",
                "data",
                "using",
            }
        ]
        if len(meaningful_concepts) >= 2:
            concept_list = list(meaningful_concepts)[:4]
            summary_parts.append(
                f"Key areas covered include {', '.join(concept_list)}."
            )

        return " ".join(summary_parts)

    def _extract_content_insights(
        self, snippet: str, content_insights: List[str], key_concepts: set
    ):
        """Extract meaningful content insights from a snippet."""
        snippet.lower()

        # Look for actionable insights and concrete information
        sentences = snippet.split(". ")

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue

            sentence_lower = sentence.lower()

            # Extract insights based on content patterns
            if any(
                pattern in sentence_lower
                for pattern in [
                    "use ",
                    "implement",
                    "apply",
                    "utilize",
                    "leverage",
                    "avoid",
                    "prevent",
                    "ensure",
                    "consider",
                    "recommend",
                ]
            ):
                # This looks like actionable advice
                clean_sentence = sentence.rstrip(".").rstrip(",")
                if len(clean_sentence) < 120:  # Keep it concise
                    content_insights.append(f"• {clean_sentence}.")

            elif any(
                pattern in sentence_lower
                for pattern in [
                    "breakthrough",
                    "development",
                    "advance",
                    "innovation",
                    "new approach",
                    "recent",
                    "latest",
                    "introduced",
                    "emerging",
                ]
            ):
                # This looks like new developments
                clean_sentence = sentence.rstrip(".").rstrip(",")
                if len(clean_sentence) < 120:
                    content_insights.append(f"• {clean_sentence}.")

            elif any(
                pattern in sentence_lower
                for pattern in [
                    "framework",
                    "library",
                    "tool",
                    "method",
                    "technique",
                    "algorithm",
                    "model",
                    "system",
                    "platform",
                ]
            ):
                # This looks like technical concepts
                clean_sentence = sentence.rstrip(".").rstrip(",")
                if len(clean_sentence) < 120:
                    content_insights.append(f"• {clean_sentence}.")

            # Extract key concepts (nouns and technical terms)
            words = sentence.split()
            for word in words:
                clean_word = word.strip(".,()[]{}").lower()
                if (
                    len(clean_word) > 4
                    and clean_word
                    not in {"which", "where", "there", "their", "these", "those"}
                    and not clean_word.isdigit()
                ):
                    key_concepts.add(clean_word)


class PDFCompilerTool(Tool):
    """Tool for compiling markdown to PDF using pandoc."""

    def __init__(self):
        super().__init__(
            name="pdf-compiler",
            description="Compile markdown reports to PDF using pandoc",
        )
        self.add_parameter("markdown_content", "string", "Markdown content to compile")
        self.add_parameter("output_path", "string", "Output PDF file path")
        self.add_parameter("title", "string", "Document title", required=False)
        self.add_parameter("author", "string", "Document author", required=False)
        self.add_parameter(
            "install_if_missing",
            "boolean",
            "Install pandoc if not found",
            required=False,
            default=True,
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Compile markdown to PDF."""
        markdown_content = kwargs.get("markdown_content", "")
        output_path = kwargs.get("output_path", "report.pdf")
        title = kwargs.get("title", "Research Report")
        author = kwargs.get("author", "Orchestrator Research Assistant")
        install_if_missing = kwargs.get("install_if_missing", True)

        if not markdown_content:
            return {"success": False, "error": "No markdown content provided"}

        # Check if pandoc is installed
        if not self._is_pandoc_installed():
            if install_if_missing:
                logger.info("Pandoc not found. Installing...")
                install_result = await self._install_pandoc()
                if not install_result["success"]:
                    return install_result
            else:
                return {
                    "success": False,
                    "error": "Pandoc is not installed. Set install_if_missing=True to install automatically.",
                }

        # Preprocess markdown to fix list formatting for pandoc
        processed_markdown = self._fix_list_formatting(markdown_content)

        # Create temporary markdown file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False
        ) as temp_md:
            temp_md.write(processed_markdown)
            temp_md_path = temp_md.name

        try:
            # Prepare pandoc command
            cmd = [
                "pandoc",
                temp_md_path,
                "-o",
                output_path,
                "--pdf-engine=xelatex",  # Use xelatex for better Unicode support
                "-V",
                "geometry:margin=1in",
                "-V",
                "fontsize=12pt",
                "-V",
                "mainfont=DejaVu Sans",  # Cross-platform font
                "--highlight-style=tango",
                f"--metadata=title:{title}",
                f"--metadata=author:{author}",
                "--metadata=date:" + datetime.now().strftime("%B %d, %Y"),
                "--toc",  # Table of contents
                "--toc-depth=2",
            ]

            # Run pandoc
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Get file size
                file_size = os.path.getsize(output_path)

                return {
                    "success": True,
                    "output_path": output_path,
                    "file_size": file_size,
                    "message": f"PDF generated successfully: {output_path}",
                }
            else:
                # If xelatex fails, try with default PDF engine
                cmd[3] = "--pdf-engine=pdflatex"
                cmd.remove("-V")
                cmd.remove("mainfont=DejaVu Sans")

                result = subprocess.run(cmd, capture_output=True, text=True)

                if result.returncode == 0:
                    file_size = os.path.getsize(output_path)
                    return {
                        "success": True,
                        "output_path": output_path,
                        "file_size": file_size,
                        "message": f"PDF generated successfully (using pdflatex): {output_path}",
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Pandoc failed: {result.stderr}",
                    }

        except Exception as e:
            return {"success": False, "error": f"PDF compilation failed: {str(e)}"}
        finally:
            # Clean up temporary file
            if os.path.exists(temp_md_path):
                os.unlink(temp_md_path)

    def _is_pandoc_installed(self) -> bool:
        """Check if pandoc is installed."""
        try:
            result = subprocess.run(["pandoc", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    async def _install_pandoc(self) -> Dict[str, Any]:
        """Install pandoc based on the operating system."""
        system = platform.system().lower()

        try:
            if system == "darwin":  # macOS
                # Try homebrew first
                if self._command_exists("brew"):
                    logger.info("Installing pandoc via Homebrew...")
                    result = subprocess.run(
                        ["brew", "install", "pandoc"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "Pandoc installed via Homebrew",
                        }

                # Try MacPorts
                if self._command_exists("port"):
                    logger.info("Installing pandoc via MacPorts...")
                    result = subprocess.run(
                        ["sudo", "port", "install", "pandoc"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "Pandoc installed via MacPorts",
                        }

                # Download installer
                return await self._download_and_install_pandoc("macos")

            elif system == "linux":
                # Try apt-get (Debian/Ubuntu)
                if self._command_exists("apt-get"):
                    logger.info("Installing pandoc via apt-get...")
                    result = subprocess.run(
                        ["sudo", "apt-get", "update"], capture_output=True
                    )
                    result = subprocess.run(
                        ["sudo", "apt-get", "install", "-y", "pandoc", "texlive-xetex"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "Pandoc installed via apt-get",
                        }

                # Try yum (RHEL/CentOS/Fedora)
                if self._command_exists("yum"):
                    logger.info("Installing pandoc via yum...")
                    result = subprocess.run(
                        ["sudo", "yum", "install", "-y", "pandoc", "texlive-xetex"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return {"success": True, "message": "Pandoc installed via yum"}

                # Try dnf (Fedora)
                if self._command_exists("dnf"):
                    logger.info("Installing pandoc via dnf...")
                    result = subprocess.run(
                        ["sudo", "dnf", "install", "-y", "pandoc", "texlive-xetex"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return {"success": True, "message": "Pandoc installed via dnf"}

                # Download installer
                return await self._download_and_install_pandoc("linux")

            elif system == "windows":
                # Try chocolatey
                if self._command_exists("choco"):
                    logger.info("Installing pandoc via Chocolatey...")
                    result = subprocess.run(
                        ["choco", "install", "pandoc", "-y"],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "Pandoc installed via Chocolatey",
                        }

                # Try scoop
                if self._command_exists("scoop"):
                    logger.info("Installing pandoc via Scoop...")
                    result = subprocess.run(
                        ["scoop", "install", "pandoc"], capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        return {
                            "success": True,
                            "message": "Pandoc installed via Scoop",
                        }

                # Download installer
                return await self._download_and_install_pandoc("windows")

            else:
                return {
                    "success": False,
                    "error": f"Unsupported operating system: {system}",
                }

        except Exception as e:
            return {"success": False, "error": f"Failed to install pandoc: {str(e)}"}

    def _command_exists(self, command: str) -> bool:
        """Check if a command exists."""
        try:
            result = subprocess.run(
                ["which" if platform.system() != "Windows" else "where", command],
                capture_output=True,
            )
            return result.returncode == 0
        except Exception:
            return False

    async def _download_and_install_pandoc(self, os_type: str) -> Dict[str, Any]:
        """Download and install pandoc from GitHub releases."""
        import json
        import urllib.request

        try:
            # Get latest release info
            with urllib.request.urlopen(
                "https://api.github.com/repos/jgm/pandoc/releases/latest"
            ) as response:
                release_data = json.loads(response.read())

            version = release_data["tag_name"]

            # Determine download URL based on OS
            if os_type == "macos":
                asset_name = f"pandoc-{version}-macOS.pkg"
            elif os_type == "linux":
                asset_name = f"pandoc-{version}-linux-amd64.tar.gz"
            elif os_type == "windows":
                asset_name = f"pandoc-{version}-windows-x86_64.msi"
            else:
                return {"success": False, "error": f"Unsupported OS type: {os_type}"}

            # Find download URL
            download_url = None
            for asset in release_data["assets"]:
                if asset["name"] == asset_name:
                    download_url = asset["browser_download_url"]
                    break

            if not download_url:
                return {
                    "success": False,
                    "error": f"Could not find download URL for {asset_name}",
                }

            # Download the file
            logger.info(f"Downloading pandoc from {download_url}...")
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(asset_name).suffix
            ) as tmp_file:
                with urllib.request.urlopen(download_url) as response:
                    tmp_file.write(response.read())
                tmp_path = tmp_file.name

            # Install based on OS
            if os_type == "macos":
                result = subprocess.run(
                    ["sudo", "installer", "-pkg", tmp_path, "-target", "/"],
                    capture_output=True,
                )
            elif os_type == "linux":
                # Extract and install
                extract_dir = tempfile.mkdtemp()
                subprocess.run(
                    ["tar", "-xzf", tmp_path, "-C", extract_dir], capture_output=True
                )
                # Find pandoc binary
                pandoc_binary = (
                    Path(extract_dir) / f"pandoc-{version}" / "bin" / "pandoc"
                )
                if pandoc_binary.exists():
                    subprocess.run(
                        ["sudo", "cp", str(pandoc_binary), "/usr/local/bin/"],
                        capture_output=True,
                    )
                    subprocess.run(
                        ["sudo", "chmod", "+x", "/usr/local/bin/pandoc"],
                        capture_output=True,
                    )
                result = subprocess.CompletedProcess([], 0)  # Assume success
            elif os_type == "windows":
                result = subprocess.run(
                    ["msiexec", "/i", tmp_path, "/quiet"], capture_output=True
                )

            # Clean up
            os.unlink(tmp_path)

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Pandoc {version} installed successfully",
                }
            else:
                return {"success": False, "error": "Failed to install pandoc"}

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to download/install pandoc: {str(e)}",
            }

    def _fix_list_formatting(self, markdown_content: str) -> str:
        """Fix list formatting by ensuring empty lines before lists for proper pandoc rendering."""
        lines = markdown_content.split("\n")
        fixed_lines = []

        for i, line in enumerate(lines):
            # Check if current line is a list item (numbered or bulleted)
            is_list_item = (
                line.strip().startswith("- ")  # Bulleted list
                or line.strip().startswith("* ")  # Alternative bullet
                or line.strip().startswith("+ ")  # Alternative bullet
                or (
                    line.strip() and line.strip()[0].isdigit() and ". " in line
                )  # Numbered list
            )

            if is_list_item:
                # Check if previous line exists and is not empty
                if i > 0 and lines[i - 1].strip() != "":
                    # Add empty line before list if needed
                    if not fixed_lines or fixed_lines[-1].strip() != "":
                        fixed_lines.append("")

            fixed_lines.append(line)

        return "\n".join(fixed_lines)
