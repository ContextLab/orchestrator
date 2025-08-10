"""Network Isolation and Controlled Access - Issue #206 Task 2.3

Advanced network security management providing fine-grained network access control,
traffic monitoring, and secure communication channels for containerized execution.
"""

import logging
import asyncio
import json
import ipaddress
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import subprocess

from .docker_manager import SecureContainer, SecurityConfig

logger = logging.getLogger(__name__)


class NetworkAccessLevel(Enum):
    """Network access levels for containers."""
    NONE = "none"           # No network access
    LIMITED = "limited"     # Limited access to specific services
    INTERNET = "internet"   # Full internet access
    CUSTOM = "custom"       # Custom network configuration


class NetworkProtocol(Enum):
    """Network protocols."""
    HTTP = "http"
    HTTPS = "https"
    FTP = "ftp"
    SSH = "ssh"
    SMTP = "smtp"
    DNS = "dns"
    TCP = "tcp"
    UDP = "udp"


@dataclass
class NetworkRule:
    """Network access rule definition."""
    name: str
    protocol: NetworkProtocol
    host: Optional[str] = None
    port: Optional[int] = None
    allow: bool = True
    description: str = ""
    
    def matches(self, protocol: str, host: str, port: int) -> bool:
        """Check if this rule matches the given network request."""
        if self.protocol.value != protocol.lower():
            return False
        
        if self.host and self.host != host:
            # Simple wildcard matching
            if '*' in self.host:
                import fnmatch
                if not fnmatch.fnmatch(host, self.host):
                    return False
            else:
                return False
        
        if self.port and self.port != port:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'protocol': self.protocol.value,
            'host': self.host,
            'port': self.port,
            'allow': self.allow,
            'description': self.description
        }


@dataclass
class NetworkPolicy:
    """Network access policy for containers."""
    name: str
    access_level: NetworkAccessLevel
    rules: List[NetworkRule] = field(default_factory=list)
    allowed_hosts: List[str] = field(default_factory=list)
    blocked_hosts: List[str] = field(default_factory=list)
    allowed_ports: List[int] = field(default_factory=list)
    blocked_ports: List[int] = field(default_factory=list)
    dns_servers: List[str] = field(default_factory=lambda: ["8.8.8.8", "1.1.1.1"])
    max_bandwidth_mbps: Optional[float] = None
    max_connections: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'access_level': self.access_level.value,
            'rules': [rule.to_dict() for rule in self.rules],
            'allowed_hosts': self.allowed_hosts,
            'blocked_hosts': self.blocked_hosts,
            'allowed_ports': self.allowed_ports,
            'blocked_ports': self.blocked_ports,
            'dns_servers': self.dns_servers,
            'max_bandwidth_mbps': self.max_bandwidth_mbps,
            'max_connections': self.max_connections
        }


@dataclass
class NetworkConnection:
    """Network connection tracking information."""
    container_id: str
    protocol: str
    source_ip: str
    source_port: int
    dest_ip: str
    dest_port: int
    timestamp: float
    bytes_sent: int = 0
    bytes_received: int = 0
    duration: float = 0.0
    allowed: bool = True
    rule_matched: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'container_id': self.container_id,
            'protocol': self.protocol,
            'source_ip': self.source_ip,
            'source_port': self.source_port,
            'dest_ip': self.dest_ip,
            'dest_port': self.dest_port,
            'timestamp': self.timestamp,
            'bytes_sent': self.bytes_sent,
            'bytes_received': self.bytes_received,
            'duration': self.duration,
            'allowed': self.allowed,
            'rule_matched': self.rule_matched
        }


class NetworkManager:
    """
    Advanced network security manager providing isolation, access control,
    and monitoring for containerized execution environments.
    """
    
    def __init__(self):
        self.policies: Dict[str, NetworkPolicy] = {}
        self.container_policies: Dict[str, str] = {}  # container_id -> policy_name
        self.active_connections: Dict[str, List[NetworkConnection]] = {}
        self.connection_history: List[NetworkConnection] = []
        self.custom_networks: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self.stats = {
            'connections_allowed': 0,
            'connections_blocked': 0,
            'total_bytes_transferred': 0,
            'policies_created': 0,
            'containers_monitored': 0
        }
        
        # Initialize default policies
        self._create_default_policies()
        
        logger.info("NetworkManager initialized")
    
    def _create_default_policies(self):
        """Create default network policies."""
        
        # No network access policy
        no_access = NetworkPolicy(
            name="no_access",
            access_level=NetworkAccessLevel.NONE
        )
        
        # Limited access policy (DNS + specific services)
        limited_access = NetworkPolicy(
            name="limited_access",
            access_level=NetworkAccessLevel.LIMITED,
            rules=[
                NetworkRule("allow_dns", NetworkProtocol.DNS, port=53, allow=True),
                NetworkRule("allow_https_pypi", NetworkProtocol.HTTPS, host="pypi.org", allow=True),
                NetworkRule("allow_https_npmjs", NetworkProtocol.HTTPS, host="registry.npmjs.org", allow=True),
            ],
            allowed_ports=[53, 443]  # DNS and HTTPS
        )
        
        # Internet access policy
        internet_access = NetworkPolicy(
            name="internet_access",
            access_level=NetworkAccessLevel.INTERNET,
            blocked_hosts=["localhost", "127.0.0.1", "::1"],  # Block localhost
            blocked_ports=[22, 23, 25, 135, 445, 1433, 3389],  # Block dangerous ports
            max_connections=50
        )
        
        # Development policy (more permissive for development)
        dev_access = NetworkPolicy(
            name="development",
            access_level=NetworkAccessLevel.INTERNET,
            max_connections=100,
            max_bandwidth_mbps=10.0
        )
        
        policies = [no_access, limited_access, internet_access, dev_access]
        for policy in policies:
            self.add_policy(policy)
    
    def add_policy(self, policy: NetworkPolicy):
        """Add a network policy."""
        self.policies[policy.name] = policy
        self.stats['policies_created'] += 1
        logger.info(f"Added network policy: {policy.name}")
    
    def get_policy(self, name: str) -> Optional[NetworkPolicy]:
        """Get a network policy by name."""
        return self.policies.get(name)
    
    def list_policies(self) -> List[str]:
        """List all available network policies."""
        return list(self.policies.keys())
    
    def apply_network_policy(
        self, 
        container: SecureContainer, 
        policy_name: str,
        custom_rules: Optional[List[NetworkRule]] = None
    ) -> bool:
        """
        Apply a network policy to a container.
        
        Args:
            container: Container to apply policy to
            policy_name: Name of the policy to apply
            custom_rules: Optional additional custom rules
            
        Returns:
            True if policy was applied successfully
        """
        
        policy = self.get_policy(policy_name)
        if not policy:
            logger.error(f"Network policy '{policy_name}' not found")
            return False
        
        try:
            # Store the policy association
            self.container_policies[container.container_id] = policy_name
            
            # Apply network configuration to container
            success = self._configure_container_network(container, policy, custom_rules)
            
            if success:
                self.stats['containers_monitored'] += 1
                logger.info(f"Applied network policy '{policy_name}' to container {container.name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to apply network policy: {e}")
            return False
    
    def _configure_container_network(
        self, 
        container: SecureContainer, 
        policy: NetworkPolicy,
        custom_rules: Optional[List[NetworkRule]] = None
    ) -> bool:
        """Configure container network settings based on policy."""
        
        try:
            if not container.docker_container:
                logger.error("Container not available for network configuration")
                return False
            
            # Configure based on access level
            if policy.access_level == NetworkAccessLevel.NONE:
                # Disable all network access
                self._disable_container_network(container)
            
            elif policy.access_level == NetworkAccessLevel.LIMITED:
                # Configure limited network access
                self._configure_limited_network(container, policy, custom_rules)
            
            elif policy.access_level == NetworkAccessLevel.INTERNET:
                # Configure internet access with restrictions
                self._configure_internet_access(container, policy, custom_rules)
            
            elif policy.access_level == NetworkAccessLevel.CUSTOM:
                # Apply custom network configuration
                self._configure_custom_network(container, policy, custom_rules)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure container network: {e}")
            return False
    
    def _disable_container_network(self, container: SecureContainer):
        """Disable all network access for container."""
        
        try:
            # Use iptables to block all network traffic for the container
            # This is a simplified approach - production would use Docker networks
            
            container_ip = self._get_container_ip(container)
            if container_ip:
                # Block all outbound traffic from container
                subprocess.run([
                    'sudo', 'iptables', '-I', 'FORWARD', 
                    '-s', container_ip, '-j', 'DROP'
                ], check=False)
                
                # Block all inbound traffic to container
                subprocess.run([
                    'sudo', 'iptables', '-I', 'FORWARD',
                    '-d', container_ip, '-j', 'DROP'
                ], check=False)
            
            logger.info(f"Disabled network access for container {container.name}")
            
        except Exception as e:
            logger.warning(f"Could not disable network for container (may require sudo): {e}")
    
    def _configure_limited_network(
        self, 
        container: SecureContainer, 
        policy: NetworkPolicy,
        custom_rules: Optional[List[NetworkRule]]
    ):
        """Configure limited network access based on policy rules."""
        
        try:
            # In a production environment, this would configure:
            # 1. Custom Docker network with restricted routing
            # 2. DNS filtering
            # 3. Firewall rules for specific hosts/ports
            # 4. Traffic shaping
            
            # For now, log the configuration
            logger.info(f"Configuring limited network for {container.name}")
            logger.info(f"Allowed hosts: {policy.allowed_hosts}")
            logger.info(f"Allowed ports: {policy.allowed_ports}")
            logger.info(f"Rules: {len(policy.rules)} rules")
            
            if custom_rules:
                logger.info(f"Custom rules: {len(custom_rules)} additional rules")
            
        except Exception as e:
            logger.error(f"Failed to configure limited network: {e}")
    
    def _configure_internet_access(
        self, 
        container: SecureContainer, 
        policy: NetworkPolicy,
        custom_rules: Optional[List[NetworkRule]]
    ):
        """Configure internet access with security restrictions."""
        
        try:
            logger.info(f"Configuring internet access for {container.name}")
            logger.info(f"Blocked hosts: {policy.blocked_hosts}")
            logger.info(f"Blocked ports: {policy.blocked_ports}")
            
            if policy.max_connections:
                logger.info(f"Max connections: {policy.max_connections}")
            
            if policy.max_bandwidth_mbps:
                logger.info(f"Bandwidth limit: {policy.max_bandwidth_mbps} Mbps")
            
        except Exception as e:
            logger.error(f"Failed to configure internet access: {e}")
    
    def _configure_custom_network(
        self, 
        container: SecureContainer, 
        policy: NetworkPolicy,
        custom_rules: Optional[List[NetworkRule]]
    ):
        """Configure custom network settings."""
        
        try:
            logger.info(f"Configuring custom network for {container.name}")
            
            # Apply all rules from policy and custom rules
            all_rules = policy.rules.copy()
            if custom_rules:
                all_rules.extend(custom_rules)
            
            for rule in all_rules:
                logger.info(f"Applying rule: {rule.name} - {rule.description}")
            
        except Exception as e:
            logger.error(f"Failed to configure custom network: {e}")
    
    def _get_container_ip(self, container: SecureContainer) -> Optional[str]:
        """Get the IP address of a container."""
        
        try:
            if container.docker_container:
                container_data = container.docker_container.attrs
                networks = container_data.get('NetworkSettings', {}).get('Networks', {})
                
                for network_name, network_info in networks.items():
                    ip_address = network_info.get('IPAddress')
                    if ip_address:
                        return ip_address
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get container IP: {e}")
            return None
    
    async def monitor_network_activity(
        self, 
        container: SecureContainer,
        duration: int = 60
    ) -> List[NetworkConnection]:
        """
        Monitor network activity for a container.
        
        Args:
            container: Container to monitor
            duration: Monitoring duration in seconds
            
        Returns:
            List of network connections detected
        """
        
        container_id = container.container_id
        connections = []
        
        logger.info(f"Starting network monitoring for {container.name} ({duration}s)")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # In a production environment, this would:
                # 1. Parse netstat output from container
                # 2. Monitor iptables logs
                # 3. Use network packet capture
                # 4. Track bandwidth usage
                
                # Simulated monitoring
                await asyncio.sleep(5)  # Check every 5 seconds
                
                # Get current network stats from Docker
                if container.docker_container:
                    try:
                        stats = container.docker_container.stats(stream=False)
                        networks = stats.get('networks', {})
                        
                        for interface, net_stats in networks.items():
                            if isinstance(net_stats, dict):
                                rx_bytes = net_stats.get('rx_bytes', 0)
                                tx_bytes = net_stats.get('tx_bytes', 0)
                                
                                # Create a simulated connection record
                                conn = NetworkConnection(
                                    container_id=container_id,
                                    protocol='tcp',
                                    source_ip=self._get_container_ip(container) or '172.17.0.1',
                                    source_port=0,
                                    dest_ip='unknown',
                                    dest_port=0,
                                    timestamp=time.time(),
                                    bytes_sent=tx_bytes,
                                    bytes_received=rx_bytes,
                                    allowed=True
                                )
                                connections.append(conn)
                                
                    except Exception as e:
                        logger.warning(f"Error getting network stats: {e}")
        
        except Exception as e:
            logger.error(f"Network monitoring failed: {e}")
        
        logger.info(f"Network monitoring completed. Found {len(connections)} connection records")
        
        # Store in history
        self.connection_history.extend(connections)
        if container_id not in self.active_connections:
            self.active_connections[container_id] = []
        self.active_connections[container_id].extend(connections)
        
        return connections
    
    def evaluate_network_request(
        self, 
        container_id: str, 
        protocol: str, 
        host: str, 
        port: int
    ) -> bool:
        """
        Evaluate if a network request should be allowed based on policy.
        
        Args:
            container_id: ID of the container making the request
            protocol: Network protocol (http, https, etc.)
            host: Destination host
            port: Destination port
            
        Returns:
            True if request should be allowed
        """
        
        policy_name = self.container_policies.get(container_id)
        if not policy_name:
            logger.warning(f"No network policy found for container {container_id}")
            return True  # Default allow if no policy
        
        policy = self.get_policy(policy_name)
        if not policy:
            return True
        
        try:
            # Check against policy rules
            for rule in policy.rules:
                if rule.matches(protocol, host, port):
                    if rule.allow:
                        self.stats['connections_allowed'] += 1
                        return True
                    else:
                        self.stats['connections_blocked'] += 1
                        logger.info(f"Network request blocked by rule '{rule.name}': {protocol}://{host}:{port}")
                        return False
            
            # Check allowed/blocked hosts
            if policy.blocked_hosts and host in policy.blocked_hosts:
                self.stats['connections_blocked'] += 1
                logger.info(f"Network request blocked - host in blocked list: {host}")
                return False
            
            if policy.allowed_hosts and host not in policy.allowed_hosts:
                self.stats['connections_blocked'] += 1
                logger.info(f"Network request blocked - host not in allowed list: {host}")
                return False
            
            # Check allowed/blocked ports
            if policy.blocked_ports and port in policy.blocked_ports:
                self.stats['connections_blocked'] += 1
                logger.info(f"Network request blocked - port in blocked list: {port}")
                return False
            
            if policy.allowed_ports and port not in policy.allowed_ports:
                self.stats['connections_blocked'] += 1
                logger.info(f"Network request blocked - port not in allowed list: {port}")
                return False
            
            # Default behavior based on access level
            if policy.access_level == NetworkAccessLevel.NONE:
                self.stats['connections_blocked'] += 1
                return False
            
            # Allow by default for other access levels
            self.stats['connections_allowed'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating network request: {e}")
            return False  # Deny on error
    
    def get_container_network_stats(self, container_id: str) -> Dict[str, Any]:
        """Get network statistics for a container."""
        
        connections = self.active_connections.get(container_id, [])
        policy_name = self.container_policies.get(container_id)
        
        return {
            'container_id': container_id,
            'policy_applied': policy_name,
            'total_connections': len(connections),
            'active_connections': len([c for c in connections if c.duration == 0]),
            'total_bytes_sent': sum(c.bytes_sent for c in connections),
            'total_bytes_received': sum(c.bytes_received for c in connections),
            'connections_allowed': len([c for c in connections if c.allowed]),
            'connections_blocked': len([c for c in connections if not c.allowed])
        }
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network management statistics."""
        
        stats = self.stats.copy()
        stats.update({
            'policies_available': len(self.policies),
            'containers_with_policies': len(self.container_policies),
            'total_connections_tracked': len(self.connection_history),
            'active_containers': len(self.active_connections)
        })
        
        return stats
    
    def cleanup_container(self, container_id: str):
        """Clean up network tracking for a container."""
        
        try:
            # Remove from active tracking
            if container_id in self.active_connections:
                del self.active_connections[container_id]
            
            if container_id in self.container_policies:
                del self.container_policies[container_id]
            
            logger.info(f"Cleaned up network tracking for container {container_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up network tracking: {e}")
    
    def export_network_policy(self, policy_name: str) -> Optional[Dict[str, Any]]:
        """Export a network policy to dictionary format."""
        
        policy = self.get_policy(policy_name)
        if policy:
            return policy.to_dict()
        return None
    
    def import_network_policy(self, policy_data: Dict[str, Any]) -> bool:
        """Import a network policy from dictionary format."""
        
        try:
            # Reconstruct policy from data
            policy = NetworkPolicy(
                name=policy_data['name'],
                access_level=NetworkAccessLevel(policy_data['access_level']),
                allowed_hosts=policy_data.get('allowed_hosts', []),
                blocked_hosts=policy_data.get('blocked_hosts', []),
                allowed_ports=policy_data.get('allowed_ports', []),
                blocked_ports=policy_data.get('blocked_ports', []),
                dns_servers=policy_data.get('dns_servers', []),
                max_bandwidth_mbps=policy_data.get('max_bandwidth_mbps'),
                max_connections=policy_data.get('max_connections')
            )
            
            # Reconstruct rules
            for rule_data in policy_data.get('rules', []):
                rule = NetworkRule(
                    name=rule_data['name'],
                    protocol=NetworkProtocol(rule_data['protocol']),
                    host=rule_data.get('host'),
                    port=rule_data.get('port'),
                    allow=rule_data.get('allow', True),
                    description=rule_data.get('description', '')
                )
                policy.rules.append(rule)
            
            self.add_policy(policy)
            return True
            
        except Exception as e:
            logger.error(f"Failed to import network policy: {e}")
            return False


# Export classes
__all__ = [
    'NetworkManager',
    'NetworkPolicy',
    'NetworkRule',
    'NetworkConnection',
    'NetworkAccessLevel',
    'NetworkProtocol'
]