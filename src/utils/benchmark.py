from enum import Enum
import json
from dataclasses import dataclass


class FormulaType(Enum):
    VALID = 'valid'
    SATISFIABLE_NOT_VALID = 'satisfiable_not_valid'
    UNSATISFIABLE = 'unsatisfiable'
    UNKNOWN = 'unknown'

@dataclass
class BenchmarkMetadata:
    num_instances: int | None
    num_valid: int | None
    num_satisfiable_not_valid: int | None
    num_unsatisfiable: int | None
    seed: int | None


class BenchmarkFormula:
    def __init__(self, formula: str, size: int, type: FormulaType) -> None:
        self.formula = formula
        self.size = size
        self.type = type
    
    def __str__(self) -> str:
        return f"{self.formula}"


class Benchmark:
    def __init__(self, metadata: BenchmarkMetadata, formulas: list[BenchmarkFormula]):
        self.metadata = metadata
        self.formulas = formulas

        self.fill_metadata()
    

    def fill_metadata(self):
        if self.metadata.num_instances is None: self.metadata.num_instances = len(self.formulas)
        if self.metadata.num_valid is None: self.metadata.num_valid = sum(1 for f in self.formulas if f.type == FormulaType.VALID)
        if self.metadata.num_satisfiable_not_valid is None: self.metadata.num_satisfiable_not_valid = sum(1 for f in self.formulas if f.type == FormulaType.SATISFIABLE_NOT_VALID)
        if self.metadata.num_unsatisfiable is None: self.metadata.num_unsatisfiable = sum(1 for f in self.formulas if f.type == FormulaType.UNSATISFIABLE)


    def save(self, filepath: str) -> None:
        data = {
            'metadata': {
                'num_instances': self.metadata.num_instances,
                'num_valid': self.metadata.num_valid,
                'num_satisfiable_not_valid': self.metadata.num_satisfiable_not_valid,
                'num_unsatisfiable': self.metadata.num_unsatisfiable,
                'seed': self.metadata.seed
            },
            'formulas': [
                {
                    'formula': f.formula,
                    'size': f.size,
                    'type': f.type.value
                }
                for f in self.formulas
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Benchmark':
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        metadata = BenchmarkMetadata(
            num_instances=data['metadata']['num_instances'],
            num_valid=data['metadata']['num_valid'],
            num_satisfiable_not_valid=data['metadata']['num_satisfiable_not_valid'],
            num_unsatisfiable=data['metadata']['num_unsatisfiable'],
            seed=data['metadata']['seed']
        )
        
        formulas = [
            BenchmarkFormula(
                formula=f['formula'],
                size=f['size'],
                type=FormulaType(f['type'])
            )
            for f in data['formulas']
        ]
        
        return cls(metadata, formulas)
