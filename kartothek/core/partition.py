# -*- coding: utf-8 -*-


class Partition:
    def __init__(self, label, files=None, metadata=None):
        """
        An object for the internal representation of the metadata of a partition.

        This class is for internal use only

        Parameters
        ----------
        label: str
            A label identifying the partition, e.g. `partition_1` or `P=0/L=A`
        files: dict, optional
            A dictionary containing the keys of the files contained in this partition
        metadata: dict, optional
            Partition level, custom metadata
        """
        self.label = label
        self.files = files if files else {}

    def __eq__(self, other):
        if not isinstance(other, Partition):
            return False
        if self.label != other.label:
            return False
        if self.files != other.files:
            return False
        return True

    @staticmethod
    def from_dict(label, dct):
        if isinstance(dct, str):
            raise ValueError(
                "Trying to load a partition from a string. Probably the dataset file uses the multifile "
                "feature. Please load the metadata object using the DatasetMetadata.load_from_buffer "
                "method instead to resolve references to external partitions."
            )
        return Partition(label, files=dct.get("files", {}))

    def to_dict(self, version=None):
        return {"files": self.files}
