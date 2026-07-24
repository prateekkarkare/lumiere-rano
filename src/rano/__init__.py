"""rano — a volumetric-RANO longitudinal brain-MRI pipeline.

Piece 1 (this stage): a non-destructive data loader + validator. Two ingestion adapters reduce
heterogeneous sources to one lazy, space-aware case contract; a validator emits per-case
capability profiles; a router turns profiles into actions. The loader never transforms pixels.
"""

__version__ = "0.0.1"
