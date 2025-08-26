#!/usr/bin/env python3
"""
Pipeline Discovery Integration for Issue #255.

Integrates repository organization with existing pipeline discovery mechanisms
from src/orchestrator/tools/discovery.py to ensure seamless operation and
provide enhanced file discovery capabilities for the pipeline system.
"""

import sys
from pathlib import Path

# Add source directory to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from orchestrator.tools.discovery import ToolDiscoveryEngine
from repository_scanner import RepositoryScanner


class EnhancedDiscoveryEngine(ToolDiscoveryEngine):
    """Enhanced discovery engine with repository organization awareness."""
    
    def __init__(self, tool_registry=None, root_path="."):
        super().__init__(tool_registry)
        self.root_path = Path(root_path).resolve()
        self.repo_scanner = RepositoryScanner(str(self.root_path))
        
        # Add organization-aware patterns
        self.organization_patterns = {
            r"organize.*repository|clean.*repository|cleanup.*files": {
                "tools": ["repository-organizer"],
                "confidence": 0.95,
                "parameters": {"action": "organize"},
            },
            r"scan.*files|discover.*files|analyze.*structure": {
                "tools": ["repository-scanner"],
                "confidence": 0.90,
                "parameters": {"action": "scan"},
            },
            r"validate.*safety|check.*safety|backup.*files": {
                "tools": ["safety-validator"],
                "confidence": 0.88,
                "parameters": {"action": "validate"},
            }
        }
        
        # Extend existing patterns
        self.action_patterns.update(self.organization_patterns)
        
        # Add organization-related semantic mappings
        organization_semantics = {
            "organize": ["repository-organizer"],
            "cleanup": ["repository-organizer", "safety-validator"],
            "structure": ["repository-scanner"],
            "categorize": ["repository-scanner"],
            "backup": ["safety-validator"],
            "validate": ["safety-validator", "validation"],
            "safety": ["safety-validator"]
        }
        
        for word, tools in organization_semantics.items():
            if word in self.semantic_mappings:
                self.semantic_mappings[word].extend(tools)
            else:
                self.semantic_mappings[word] = tools
    
    def discover_pipeline_files(self, directory: str = None) -> dict:
        """Enhanced pipeline file discovery using repository scanner."""
        if directory is None:
            directory = str(self.root_path)
        
        # Use repository scanner for comprehensive discovery
        scan_results = self.repo_scanner.scan_repository()
        
        # Extract pipeline-related files
        pipeline_files = {
            'yaml_pipelines': [],
            'python_scripts': [],
            'data_files': [],
            'output_files': [],
            'test_files': [],
            'organization_status': {
                'total_files': len(scan_results['files']),
                'organized_directories': sum(1 for d in scan_results['directories'] if d.is_organized),
                'files_needing_organization': len([f for f in scan_results['files'] 
                                                  if f.subcategory in ['scattered_in_root', 'mislocated']])
            }
        }
        
        for file_info in scan_results['files']:
            file_path = str(file_info.path)
            
            if file_path.endswith('.yaml') and 'examples' in file_path:
                pipeline_files['yaml_pipelines'].append({
                    'path': file_path,
                    'category': file_info.category,
                    'organized': file_info.subcategory == 'properly_located'
                })
            elif file_path.endswith('.py') and file_info.category in ['utility_scripts', 'test_files']:
                pipeline_files['python_scripts'].append({
                    'path': file_path,
                    'category': file_info.category,
                    'organized': file_info.subcategory == 'properly_located'
                })
            elif file_info.category == 'data_files':
                pipeline_files['data_files'].append({
                    'path': file_path,
                    'size': file_info.size,
                    'organized': file_info.subcategory == 'properly_located'
                })
            elif file_info.category in ['output_files', 'examples_output']:
                pipeline_files['output_files'].append({
                    'path': file_path,
                    'category': file_info.category,
                    'organized': file_info.subcategory == 'properly_located'
                })
            elif file_info.category == 'test_files':
                pipeline_files['test_files'].append({
                    'path': file_path,
                    'organized': file_info.subcategory == 'properly_located'
                })
        
        return pipeline_files
    
    def recommend_organization_actions(self, action_description: str) -> list:
        """Recommend organization actions based on pipeline discovery needs."""
        recommendations = []
        
        # Check current organization status
        pipeline_files = self.discover_pipeline_files()
        org_status = pipeline_files['organization_status']
        
        if org_status['files_needing_organization'] > 0:
            recommendations.append({
                'action': 'organize_repository',
                'priority': 'high',
                'description': f"Organize {org_status['files_needing_organization']} scattered files",
                'tool': 'repository-organizer'
            })
        
        # Check for specific action needs
        action_lower = action_description.lower()
        
        if 'test' in action_lower:
            unorganized_tests = [f for f in pipeline_files['test_files'] if not f['organized']]
            if unorganized_tests:
                recommendations.append({
                    'action': 'organize_test_files',
                    'priority': 'medium',
                    'description': f"Organize {len(unorganized_tests)} test files for better discovery",
                    'tool': 'root-directory-organizer'
                })
        
        if 'data' in action_lower or 'pipeline' in action_lower:
            unorganized_data = [f for f in pipeline_files['data_files'] if not f['organized']]
            if unorganized_data:
                recommendations.append({
                    'action': 'organize_data_files',
                    'priority': 'medium',
                    'description': f"Organize {len(unorganized_data)} data files for pipeline efficiency",
                    'tool': 'repository-organizer'
                })
        
        if 'validate' in action_lower or 'safety' in action_lower:
            recommendations.append({
                'action': 'validate_safety',
                'priority': 'high',
                'description': "Run safety validation before any file operations",
                'tool': 'safety-validator'
            })
        
        return recommendations
    
    def get_organization_aware_tool_chain(self, action_description: str, context: dict = None) -> list:
        """Get tool chain that considers repository organization status."""
        # Get standard tool chain
        base_chain = self.get_tool_chain_for_action(action_description, context)
        
        # Check if organization is needed first
        recommendations = self.recommend_organization_actions(action_description)
        high_priority_org = [r for r in recommendations if r['priority'] == 'high']
        
        if high_priority_org:
            # Prepend organization tools to chain
            org_tools = []
            for rec in high_priority_org:
                if rec['tool'] not in [tool.tool_name for tool in base_chain]:
                    from .discovery import ToolMatch
                    org_tools.append(ToolMatch(
                        tool_name=rec['tool'],
                        confidence=0.85,
                        reasoning=f"Organization required: {rec['description']}",
                        parameters={"action": rec['action']}
                    ))
            
            return org_tools + base_chain
        
        return base_chain


def create_integrated_discovery_system(root_path: str = ".") -> EnhancedDiscoveryEngine:
    """Factory function to create an integrated discovery system."""
    return EnhancedDiscoveryEngine(root_path=root_path)


def main():
    """Demonstration of integrated discovery system."""
    print("Repository Organization + Pipeline Discovery Integration")
    print("="*60)
    
    # Create integrated system
    discovery = create_integrated_discovery_system()
    
    # Discover pipeline files
    pipeline_files = discovery.discover_pipeline_files()
    
    print(f"\nPipeline File Discovery Results:")
    print(f"YAML Pipelines: {len(pipeline_files['yaml_pipelines'])}")
    print(f"Python Scripts: {len(pipeline_files['python_scripts'])}")
    print(f"Data Files: {len(pipeline_files['data_files'])}")
    print(f"Output Files: {len(pipeline_files['output_files'])}")
    print(f"Test Files: {len(pipeline_files['test_files'])}")
    
    org_status = pipeline_files['organization_status']
    print(f"\nOrganization Status:")
    print(f"Total Files: {org_status['total_files']:,}")
    print(f"Organized Directories: {org_status['organized_directories']}")
    print(f"Files Needing Organization: {org_status['files_needing_organization']}")
    
    # Test action recommendations
    test_actions = [
        "run pipeline tests",
        "process data files", 
        "validate file safety",
        "organize repository structure"
    ]
    
    print(f"\nAction Recommendations:")
    for action in test_actions:
        recommendations = discovery.recommend_organization_actions(action)
        print(f"\nFor '{action}':")
        for rec in recommendations:
            print(f"  {rec['priority'].upper()}: {rec['description']}")


if __name__ == "__main__":
    main()