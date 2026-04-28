"""
Custom exceptions for the Multi-Omics Clinical Integration Pipeline.
"""

class PipelineError(Exception):
    """Base class for all pipeline exceptions."""
    pass

class VisualizationError(PipelineError):
    pass

class ValidationError(PipelineError):
    pass

class QualityControlError(PipelineError):
    pass

class PreprocessingError(PipelineError):
    pass

class MissingDataError(PipelineError):
    pass

class IntegrationError(PipelineError):
    pass

class SampleAlignmentError(PipelineError):
    pass

class ExportError(PipelineError):
    pass

class FHIRError(PipelineError):
    pass

class DataCollectionError(PipelineError):
    pass

class TCGAError(PipelineError):
    pass

class GEOError(PipelineError):
    pass

class ConfigurationError(PipelineError):
    pass
