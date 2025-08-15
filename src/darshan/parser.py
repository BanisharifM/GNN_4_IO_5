#!/usr/bin/env python
"""
Darshan log parser for extracting I/O counters
Supports text-based Darshan parser output for older versions
"""

import subprocess
import logging
from pathlib import Path
from typing import Dict, Optional, Union
import re

logger = logging.getLogger(__name__)


class DarshanParser:
    """Parse Darshan logs and extract I/O counters"""
    
    def __init__(self, darshan_parser_path: Optional[str] = None):
        """Initialize Darshan parser"""
        self.darshan_parser_path = darshan_parser_path or self._find_darshan_parser()
        if not self.darshan_parser_path:
            raise RuntimeError("Cannot find darshan-parser. Please install Darshan or provide path.")
        
        logger.info(f"Using darshan-parser at: {self.darshan_parser_path}")
    
    def _find_darshan_parser(self) -> Optional[str]:
        """Find darshan-parser in system PATH or common locations"""
        common_paths = [
            "~/.conda/envs/ior_env/bin/darshan-parser",
            "~/darshan-patched-install/bin/darshan-parser",
            "/usr/local/bin/darshan-parser",
            "/usr/bin/darshan-parser"
        ]
        
        for path in common_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                return str(expanded_path)
        
        return None
    
    def parse_darshan_log(self, log_path: Union[str, Path]) -> Dict:
        """Parse a Darshan log file"""
        log_path = Path(log_path)
        
        if not log_path.exists():
            raise FileNotFoundError(f"Darshan log not found: {log_path}")
        
        # Parse using text output
        return self._parse_native_log_text(log_path)
    
    def _parse_native_log_text(self, log_path: Path) -> Dict:
        """Parse native Darshan log using text output"""
        logger.info(f"Parsing Darshan log: {log_path}")
        
        # Run darshan-parser with --all flag
        cmd = [self.darshan_parser_path, '--all', str(log_path)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return self._parse_text_output(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to parse Darshan log: {e.stderr}")
            raise
    
    def _parse_text_output(self, text_output: str) -> Dict:
        """Parse text output from darshan-parser"""
        parsed_data = {
            "header": {
                "nprocs": 1,
                "walltime": 1.0,
                "jobid": "unknown"
            },
            "records": {
                "POSIX": {}
            }
        }
        
        lines = text_output.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('#'):
                continue
            
            # Parse POSIX counters (format: POSIX <tab> rank <tab> file <tab> counter <tab> value)
            if '\tPOSIX\t' in line or line.startswith('POSIX\t'):
                parts = line.split('\t')
                if len(parts) >= 5:
                    module = parts[0]
                    rank = parts[1]
                    file_id = parts[2]
                    counter_name = parts[3]
                    value_str = parts[4]
                    
                    if file_id not in parsed_data["records"]["POSIX"]:
                        parsed_data["records"]["POSIX"][file_id] = {"counters": {}}
                    
                    try:
                        # Convert to appropriate type
                        if '.' in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except:
                        value = 0
                    
                    parsed_data["records"]["POSIX"][file_id]["counters"][counter_name] = value
        
        return parsed_data
    
    def extract_posix_counters(self, darshan_data: Dict) -> Dict[str, float]:
        """Extract POSIX counters from parsed Darshan data"""
        posix_counters = {}
        
        if "records" in darshan_data and "POSIX" in darshan_data["records"]:
            posix_records = darshan_data["records"]["POSIX"]
            
            # Aggregate counters across all files
            for file_id, file_data in posix_records.items():
                if "counters" in file_data:
                    for counter_name, value in file_data["counters"].items():
                        if counter_name not in posix_counters:
                            posix_counters[counter_name] = 0
                        posix_counters[counter_name] += value
        
        return posix_counters
    
    def extract_lustre_info(self, darshan_data: Dict) -> Dict[str, float]:
        """Extract Lustre stripe information"""
        return {
            "LUSTRE_STRIPE_SIZE": 1048576,  # Default 1MB
            "LUSTRE_STRIPE_WIDTH": 1        # Default 1 OST
        }
    
    def extract_job_info(self, darshan_data: Dict) -> Dict:
        """Extract job-level information"""
        return darshan_data.get("header", {
            "nprocs": 1,
            "runtime": 1.0,
            "jobid": "unknown"
        })