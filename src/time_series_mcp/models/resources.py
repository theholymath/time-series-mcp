"""
Resource Manager for Time Series MCP
====================================
Manages generated artifacts (plots, CSVs, reports) as MCP Resources.
"""

import os
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ResourceType(Enum):
    """Types of resources that can be managed."""
    PLOT = "plot"
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    REPORT = "report"


@dataclass
class ManagedResource:
    """A managed resource (file/artifact)."""
    uri: str                          # e.g., "ts://plots/forecast_main"
    name: str                         # Human-readable name
    resource_type: ResourceType
    description: str
    file_path: Optional[str] = None   # Actual file path on disk (if saved)
    base64_data: Optional[str] = None # Base64 data (for images)
    mime_type: str = "application/octet-stream"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "uri": self.uri,
            "name": self.name,
            "type": self.resource_type.value,
            "description": self.description,
            "file_path": self.file_path,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_data": self.base64_data is not None,
            "is_saved": self.file_path is not None and os.path.exists(self.file_path)
        }


class ResourceManager:
    """
    Manages MCP Resources for generated artifacts.
    
    URI Scheme: ts://{category}/{resource_name}
    
    Categories:
    - plots: Generated visualizations
    - data: CSV exports, processed data
    - reports: HTML/PDF reports
    """
    
    URI_SCHEME = "ts"
    
    def __init__(self, output_dir: str = "./forecast_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage of resources
        self._resources: Dict[str, ManagedResource] = {}
        
        # Category subdirectories
        self._categories = {
            "plots": self.output_dir / "plots",
            "data": self.output_dir / "data",
            "reports": self.output_dir / "reports",
        }
        
        # Create subdirectories
        for path in self._categories.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def _make_uri(self, category: str, name: str) -> str:
        """Create a URI for a resource."""
        return f"{self.URI_SCHEME}://{category}/{name}"
    
    def _parse_uri(self, uri: str) -> tuple[str, str]:
        """Parse a URI into category and name."""
        if not uri.startswith(f"{self.URI_SCHEME}://"):
            raise ValueError(f"Invalid URI scheme: {uri}")
        
        path = uri[len(f"{self.URI_SCHEME}://"):]
        parts = path.split("/", 1)
        
        if len(parts) != 2:
            raise ValueError(f"Invalid URI format: {uri}")
        
        return parts[0], parts[1]
    
    # ============ Plot Resources ============
    
    def create_plot_resource(
        self,
        name: str,
        base64_data: str,
        description: str,
        dataset_name: Optional[str] = None,
        plot_type: str = "forecast",
        auto_save: bool = True,
        file_format: str = "png"
    ) -> ManagedResource:
        """
        Create a plot resource from base64 image data.
        
        Args:
            name: Resource name (e.g., "sales_forecast_main")
            base64_data: Base64 encoded image data
            description: Human-readable description
            dataset_name: Associated dataset name
            plot_type: Type of plot (forecast, analysis, comparison, etc.)
            auto_save: Automatically save to disk
            file_format: Image format (png, svg, pdf)
        
        Returns:
            ManagedResource object
        """
        uri = self._make_uri("plots", name)
        
        # Determine mime type
        mime_types = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg"
        }
        mime_type = mime_types.get(file_format, "image/png")
        
        resource = ManagedResource(
            uri=uri,
            name=name,
            resource_type=ResourceType.PLOT,
            description=description,
            base64_data=base64_data,
            mime_type=mime_type,
            metadata={
                "dataset_name": dataset_name,
                "plot_type": plot_type,
                "format": file_format
            }
        )
        
        # Auto-save to disk
        if auto_save:
            file_path = self.save_resource_to_file(resource, file_format)
            resource.file_path = file_path
        
        # Store in memory
        self._resources[uri] = resource
        
        return resource
    
    # ============ Data Resources ============
    
    def create_data_resource(
        self,
        name: str,
        data: Any,  # DataFrame or dict
        description: str,
        data_type: str = "csv",
        dataset_name: Optional[str] = None,
        auto_save: bool = True
    ) -> ManagedResource:
        """
        Create a data resource (CSV, JSON).
        
        Args:
            name: Resource name
            data: DataFrame or dictionary to store
            description: Human-readable description
            data_type: Type of data (csv, json)
            dataset_name: Associated dataset name
            auto_save: Automatically save to disk
        
        Returns:
            ManagedResource object
        """
        import pandas as pd
        
        uri = self._make_uri("data", name)
        
        # Convert data to appropriate format
        if data_type == "csv":
            if isinstance(data, pd.DataFrame):
                content = data.to_csv(index=True)
            else:
                content = str(data)
            mime_type = "text/csv"
        else:  # json
            if isinstance(data, pd.DataFrame):
                content = data.to_json(orient='records', date_format='iso')
            else:
                content = json.dumps(data, default=str)
            mime_type = "application/json"
        
        # Encode as base64 for consistency
        base64_data = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        
        resource = ManagedResource(
            uri=uri,
            name=name,
            resource_type=ResourceType.CSV if data_type == "csv" else ResourceType.JSON,
            description=description,
            base64_data=base64_data,
            mime_type=mime_type,
            metadata={
                "dataset_name": dataset_name,
                "data_type": data_type
            }
        )
        
        if auto_save:
            file_path = self.save_resource_to_file(resource, data_type)
            resource.file_path = file_path
        
        self._resources[uri] = resource
        
        return resource
    
    # ============ Resource Management ============
    
    def get_resource(self, uri: str) -> Optional[ManagedResource]:
        """Get a resource by URI."""
        return self._resources.get(uri)
    
    def get_resource_by_name(self, name: str, category: str = "plots") -> Optional[ManagedResource]:
        """Get a resource by name and category."""
        uri = self._make_uri(category, name)
        return self._resources.get(uri)
    
    def list_resources(
        self,
        category: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
        dataset_name: Optional[str] = None
    ) -> List[ManagedResource]:
        """
        List resources with optional filtering.
        
        Args:
            category: Filter by category (plots, data, reports)
            resource_type: Filter by type
            dataset_name: Filter by associated dataset
        
        Returns:
            List of matching resources
        """
        resources = list(self._resources.values())
        
        if category:
            resources = [r for r in resources if r.uri.startswith(f"{self.URI_SCHEME}://{category}/")]
        
        if resource_type:
            resources = [r for r in resources if r.resource_type == resource_type]
        
        if dataset_name:
            resources = [r for r in resources if r.metadata.get("dataset_name") == dataset_name]
        
        return resources
    
    def delete_resource(self, uri: str, delete_file: bool = True) -> bool:
        """Delete a resource."""
        resource = self._resources.pop(uri, None)
        
        if resource and delete_file and resource.file_path:
            try:
                os.remove(resource.file_path)
            except OSError:
                pass
        
        return resource is not None
    
    def update_resource(
        self,
        uri: str,
        base64_data: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        save_to_file: bool = True
    ) -> Optional[ManagedResource]:
        """
        Update an existing resource.
        
        Args:
            uri: Resource URI
            base64_data: New base64 data (if updating content)
            description: New description
            metadata: Additional metadata to merge
            save_to_file: Save updated content to disk
        
        Returns:
            Updated resource or None if not found
        """
        resource = self._resources.get(uri)
        
        if not resource:
            return None
        
        if base64_data:
            resource.base64_data = base64_data
        
        if description:
            resource.description = description
        
        if metadata:
            resource.metadata.update(metadata)
        
        resource.updated_at = datetime.now().isoformat()
        
        # Re-save to disk if content updated
        if save_to_file and base64_data and resource.file_path:
            ext = resource.metadata.get("format", "png")
            self.save_resource_to_file(resource, ext)
        
        return resource
    
    # ============ File Operations ============
    
    def save_resource_to_file(
        self,
        resource: ManagedResource,
        extension: str
    ) -> str:
        """
        Save a resource to disk.
        
        Args:
            resource: The resource to save
            extension: File extension
        
        Returns:
            File path where saved
        """
        category, name = self._parse_uri(resource.uri)
        
        # Clean filename
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        filename = f"{safe_name}.{extension}"
        
        file_path = self._categories.get(category, self.output_dir) / filename
        
        # Decode and write
        if resource.base64_data:
            data = base64.b64decode(resource.base64_data)
            
            # Write as binary for images, text for others
            if resource.resource_type == ResourceType.PLOT:
                with open(file_path, 'wb') as f:
                    f.write(data)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(data.decode('utf-8'))
        
        resource.file_path = str(file_path)
        return str(file_path)
    
    def export_resource(
        self,
        uri: str,
        output_path: str,
        format: Optional[str] = None
    ) -> str:
        """
        Export a resource to a specific path.
        
        Args:
            uri: Resource URI
            output_path: Destination path (can include filename or just directory)
            format: Optional format conversion (for plots: png, svg, pdf)
        
        Returns:
            Path where file was saved
        """
        resource = self._resources.get(uri)
        
        if not resource:
            raise ValueError(f"Resource not found: {uri}")
        
        output_path = Path(output_path)
        
        # If output_path is a directory, add filename
        if output_path.is_dir() or not output_path.suffix:
            output_path.mkdir(parents=True, exist_ok=True)
            ext = format or resource.metadata.get("format", "png")
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in resource.name)
            output_path = output_path / f"{safe_name}.{ext}"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # For now, just copy the data
        # TODO: Add format conversion (e.g., PNG to SVG)
        if resource.base64_data:
            data = base64.b64decode(resource.base64_data)
            
            if resource.resource_type == ResourceType.PLOT:
                with open(output_path, 'wb') as f:
                    f.write(data)
            else:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(data.decode('utf-8'))
        elif resource.file_path and os.path.exists(resource.file_path):
            import shutil
            shutil.copy2(resource.file_path, output_path)
        else:
            raise ValueError(f"Resource has no data: {uri}")
        
        return str(output_path)
    
    def get_file_path(self, uri: str) -> Optional[str]:
        """Get the file path for a resource (saving if needed)."""
        resource = self._resources.get(uri)
        
        if not resource:
            return None
        
        if resource.file_path and os.path.exists(resource.file_path):
            return resource.file_path
        
        # Save to file if not already saved
        if resource.base64_data:
            ext = resource.metadata.get("format", "png")
            return self.save_resource_to_file(resource, ext)
        
        return None


# Global resource manager instance
resource_manager = ResourceManager()