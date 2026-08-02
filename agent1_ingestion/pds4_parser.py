"""
PDS4 Label Parser
==================
Parse PDS4 XML label files (.xml / .lbl) accompanying Chandrayaan-2 DFSAR
products to extract array dimensions, data types, polarization channels,
and calibration/scaling factors.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# PDS4 XML namespace
PDS4_NS = "http://pds.nasa.gov/pds4/pds/v1"
NS = {"pds": PDS4_NS}


@dataclass
class ArrayDescriptor:
    """Describes one data array within a PDS4 product."""

    name: str
    axes: int                            # Number of axes
    shape: tuple[int, ...]               # Dimension sizes (e.g., (lines, samples))
    data_type: str                       # e.g., "IEEE754MSBDouble", "ComplexLSB8"
    offset_bytes: int = 0               # Byte offset in binary file
    scaling_factor: float = 1.0
    value_offset: float = 0.0
    unit: str = ""

    @property
    def numpy_dtype(self) -> str:
        """Map PDS4 data type strings to NumPy dtype strings."""
        dtype_map = {
            "IEEE754MSBSingle": ">f4",
            "IEEE754MSBDouble": ">f8",
            "IEEE754LSBSingle": "<f4",
            "IEEE754LSBDouble": "<f8",
            "SignedMSB2": ">i2",
            "SignedMSB4": ">i4",
            "SignedLSB2": "<i2",
            "SignedLSB4": "<i4",
            "UnsignedByte": "u1",
            "UnsignedMSB2": ">u2",
            "ComplexMSB8": ">c8",
            "ComplexLSB8": "<c8",
            "ComplexMSB16": ">c16",
            "ComplexLSB16": "<c16",
        }
        return dtype_map.get(self.data_type, "<f4")


@dataclass
class ProductMetadata:
    """Parsed metadata from a PDS4 product label.

    Contains all information needed to read and interpret the binary data file.
    """

    label_path: str
    product_id: str = ""
    title: str = ""
    instrument: str = ""

    # Polarization info
    polarization_channels: list[str] = field(default_factory=list)  # e.g., ["HH", "HV", "VH", "VV"]
    band: str = ""                                                   # "L" or "S"

    # Acquisition geometry
    incidence_angle_deg: Optional[float] = None
    look_direction: str = ""

    # Data arrays
    arrays: list[ArrayDescriptor] = field(default_factory=list)

    # Calibration
    calibration_factor: float = 1.0

    # File reference
    data_file: str = ""  # Binary data file name (relative to label)

    @property
    def primary_array(self) -> Optional[ArrayDescriptor]:
        """Return the first (primary) data array."""
        return self.arrays[0] if self.arrays else None


def parse_pds4_label(label_path: str | Path) -> ProductMetadata:
    """Parse a PDS4 XML label file and extract product metadata.

    Args:
        label_path: Path to the PDS4 .xml or .lbl label file.

    Returns:
        ProductMetadata containing array descriptors, polarization info, etc.

    Raises:
        FileNotFoundError: If the label file doesn't exist.
        ET.ParseError: If the XML is malformed.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"PDS4 label not found: {label_path}")

    logger.info(f"Parsing PDS4 label: {label_path}")
    tree = ET.parse(str(label_path))
    root = tree.getroot()

    metadata = ProductMetadata(label_path=str(label_path))

    # --- Extract Identification Area ---
    ident = root.find(f".//{{{PDS4_NS}}}Identification_Area")
    if ident is not None:
        lid = ident.find(f"{{{PDS4_NS}}}logical_identifier")
        if lid is not None and lid.text:
            metadata.product_id = lid.text
        title = ident.find(f"{{{PDS4_NS}}}title")
        if title is not None and title.text:
            metadata.title = title.text

    # --- Extract Observation Area (instrument, geometry) ---
    obs = root.find(f".//{{{PDS4_NS}}}Observation_Area")
    if obs is not None:
        _parse_observation_area(obs, metadata)

    # --- Extract File Area (data arrays) ---
    file_area = root.find(f".//{{{PDS4_NS}}}File_Area_Observational")
    if file_area is not None:
        _parse_file_area(file_area, metadata)

    # --- Infer polarization from title/product_id ---
    _infer_polarization(metadata)

    logger.info(
        f"Parsed: product_id={metadata.product_id}, "
        f"arrays={len(metadata.arrays)}, "
        f"channels={metadata.polarization_channels}"
    )
    return metadata


def _parse_observation_area(obs: ET.Element, metadata: ProductMetadata) -> None:
    """Extract instrument and geometry info from Observation_Area."""
    # Instrument
    inst = obs.find(f".//{{{PDS4_NS}}}Observing_System_Component")
    if inst is not None:
        name = inst.find(f"{{{PDS4_NS}}}name")
        if name is not None and name.text:
            metadata.instrument = name.text

    # Try to find incidence angle in any geometry sub-element
    for tag in ["solar_incidence_angle", "incidence_angle"]:
        elem = obs.find(f".//{{{PDS4_NS}}}{tag}")
        if elem is not None and elem.text:
            try:
                metadata.incidence_angle_deg = float(elem.text)
            except ValueError:
                pass
            break


def _parse_file_area(file_area: ET.Element, metadata: ProductMetadata) -> None:
    """Extract data file reference and array descriptors from File_Area."""
    # Data file reference
    file_elem = file_area.find(f"{{{PDS4_NS}}}File")
    if file_elem is not None:
        fname = file_elem.find(f"{{{PDS4_NS}}}file_name")
        if fname is not None and fname.text:
            metadata.data_file = fname.text

    # Parse all Array elements (Array_2D, Array_2D_Image, Array_3D, etc.)
    for array_tag in ["Array_2D", "Array_2D_Image", "Array_3D", "Array_1D"]:
        for array_elem in file_area.findall(f"{{{PDS4_NS}}}{array_tag}"):
            descriptor = _parse_array_element(array_elem)
            if descriptor:
                metadata.arrays.append(descriptor)


def _parse_array_element(elem: ET.Element) -> Optional[ArrayDescriptor]:
    """Parse a single PDS4 Array element into an ArrayDescriptor."""
    name_elem = elem.find(f"{{{PDS4_NS}}}name")
    name = name_elem.text if name_elem is not None and name_elem.text else "unnamed"

    axes_elem = elem.find(f"{{{PDS4_NS}}}axes")
    axes = int(axes_elem.text) if axes_elem is not None and axes_elem.text else 2

    # Collect axis sizes
    shape_list = []
    for axis in elem.findall(f"{{{PDS4_NS}}}Axis_Array"):
        seq_elem = axis.find(f"{{{PDS4_NS}}}sequence_number")
        size_elem = axis.find(f"{{{PDS4_NS}}}elements")
        if size_elem is not None and size_elem.text:
            shape_list.append((
                int(seq_elem.text) if seq_elem is not None and seq_elem.text else 0,
                int(size_elem.text),
            ))
    # Sort by sequence number
    shape_list.sort(key=lambda x: x[0])
    shape = tuple(s for _, s in shape_list)

    # Data type
    dtype_elem = elem.find(f".//{{{PDS4_NS}}}data_type")
    data_type = dtype_elem.text if dtype_elem is not None and dtype_elem.text else "IEEE754LSBSingle"

    # Offset
    offset_elem = elem.find(f"{{{PDS4_NS}}}offset")
    offset = int(offset_elem.text) if offset_elem is not None and offset_elem.text else 0

    # Scaling
    scaling_elem = elem.find(f"{{{PDS4_NS}}}scaling_factor")
    scaling = float(scaling_elem.text) if scaling_elem is not None and scaling_elem.text else 1.0

    value_offset_elem = elem.find(f"{{{PDS4_NS}}}value_offset")
    value_offset = float(value_offset_elem.text) if value_offset_elem is not None and value_offset_elem.text else 0.0

    # Unit
    unit_elem = elem.find(f"{{{PDS4_NS}}}unit")
    unit = unit_elem.text if unit_elem is not None and unit_elem.text else ""

    return ArrayDescriptor(
        name=name,
        axes=axes,
        shape=shape,
        data_type=data_type,
        offset_bytes=offset,
        scaling_factor=scaling,
        value_offset=value_offset,
        unit=unit,
    )


def _infer_polarization(metadata: ProductMetadata) -> None:
    """Infer polarization channels and band from product ID / title."""
    text = (metadata.product_id + " " + metadata.title).upper()

    # Detect band
    if "L-BAND" in text or "LBAND" in text or "_L_" in text:
        metadata.band = "L"
    elif "S-BAND" in text or "SBAND" in text or "_S_" in text:
        metadata.band = "S"

    # Detect polarization channels
    pol_candidates = ["HH", "HV", "VH", "VV", "RH", "RV", "LH", "LV"]
    for pol in pol_candidates:
        if pol in text:
            metadata.polarization_channels.append(pol)

    # If none found, check array names
    if not metadata.polarization_channels:
        for arr in metadata.arrays:
            for pol in pol_candidates:
                if pol in arr.name.upper():
                    metadata.polarization_channels.append(pol)
