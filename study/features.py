"""Outcome-independent molecular/protein features and grouped allocations.

No chemical standardization, similarity cutoff, or outcome eligibility rule is
applied here. Molecular identity is the canonical isomeric SMILES of the complete
RDKit-sanitized input graph, including salts, stereochemistry, and protonation.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


DESCRIPTOR_NAMES = ("MW", "logP", "TPSA", "HBD", "HBA", "rotatable_bonds")
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class MoleculeFeatures:
    fingerprints: np.ndarray
    descriptors: np.ndarray
    identities: np.ndarray
    scaffolds: np.ndarray
    scaffold_groups: np.ndarray


def molecule_features(smiles):
    """Chiral Morgan radius 3/2048 bits, six descriptors, identity and scaffold.

    ``scaffolds`` contains exact, non-generic Murcko scaffold SMILES with
    retained scaffold stereochemistry. Acyclic molecules have empty scaffolds;
    ``scaffold_groups`` groups them by full molecular identity rather than
    putting every acyclic molecule in one group.
    """
    if isinstance(smiles, str):
        raise ValueError("Supply a sequence of SMILES strings, not one scalar string")
    smiles = list(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3, fpSize=2048, includeChirality=True)
    fingerprints = np.zeros((len(smiles), 2048), dtype=np.uint8)
    descriptors = np.empty((len(smiles), len(DESCRIPTOR_NAMES)), dtype=float)
    identities, scaffolds, groups = [], [], []
    for i, text in enumerate(smiles):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid or empty SMILES at index {i}")
        mol = Chem.MolFromSmiles(text)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid or empty SMILES at index {i}: {text!r}")
        identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=True)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fingerprints[i])
        descriptors[i] = (Descriptors.MolWt(mol), Crippen.MolLogP(mol),
                          Descriptors.TPSA(mol), Lipinski.NumHDonors(mol),
                          Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol))
        identities.append(identity)
        scaffolds.append(scaffold)
        groups.append("murcko:" + scaffold if scaffold else "acyclic:" + identity)
    return MoleculeFeatures(fingerprints, descriptors, np.asarray(identities, dtype=str),
                            np.asarray(scaffolds, dtype=str), np.asarray(groups, dtype=str))


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite two-dimensional array")
    return value


def tanimoto_kernel(x, y=None):
    """Rectangular binary-fingerprint Tanimoto; two empty fingerprints score 1."""
    x = _matrix(x, "x")
    y = x if y is None else _matrix(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Fingerprint widths differ")
    if np.any((x != 0) & (x != 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Tanimoto inputs must be binary fingerprints")
    intersection = x @ y.T
    union = x.sum(axis=1)[:, None] + y.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.ones_like(intersection), where=union != 0)


@dataclass(frozen=True)
class DescriptorRBF:
    """Population-standardized descriptor RBF fitted exclusively on supplied rows.

    The default gamma is 1 / descriptor count, a fixed technical kernel choice.
    Constant training columns use scale 1. Test rows never update these values.
    """
    mean_: np.ndarray
    scale_: np.ndarray
    gamma_: float

    @classmethod
    def fit(cls, train_descriptors, gamma=None):
        train = _matrix(train_descriptors, "train_descriptors")
        if min(train.shape) == 0:
            raise ValueError("Descriptor fitting needs at least one row and column")
        gamma = 1.0 / train.shape[1] if gamma is None else float(gamma)
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        scale = train.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        return cls(train.mean(axis=0), scale, gamma)

    def transform(self, descriptors):
        descriptors = _matrix(descriptors, "descriptors")
        if descriptors.shape[1] != self.mean_.size:
            raise ValueError("Descriptor width differs from the fitted transformer")
        return (descriptors - self.mean_) / self.scale_

    def kernel(self, x, y=None):
        x = self.transform(x)
        y = x if y is None else self.transform(y)
        distance = (np.square(x).sum(axis=1)[:, None]
                    + np.square(y).sum(axis=1)[None, :] - 2 * x @ y.T)
        return np.exp(-self.gamma_ * np.maximum(distance, 0))


@dataclass(frozen=True)
class MolecularKernel:
    """Fixed 0.5 Tanimoto + 0.5 training-standardized descriptor RBF."""
    descriptor_rbf: DescriptorRBF
    train_fingerprints: np.ndarray
    train_descriptors: np.ndarray

    @classmethod
    def fit(cls, train_fingerprints, train_descriptors):
        fps = _matrix(train_fingerprints, "train_fingerprints")
        descriptors = _matrix(train_descriptors, "train_descriptors")
        if fps.shape[0] != descriptors.shape[0]:
            raise ValueError("Fingerprint and descriptor row counts differ")
        if np.any((fps != 0) & (fps != 1)):
            raise ValueError("Fingerprints must be binary")
        return cls(DescriptorRBF.fit(descriptors), fps.copy(), descriptors.copy())

    def kernel(self, fingerprints, descriptors, other_fingerprints=None,
               other_descriptors=None):
        if (other_fingerprints is None) != (other_descriptors is None):
            raise ValueError("Both other feature arrays must be supplied together")
        chemical = tanimoto_kernel(fingerprints, other_fingerprints)
        physical = self.descriptor_rbf.kernel(descriptors, other_descriptors)
        if chemical.shape != physical.shape:
            raise ValueError("Fingerprint and descriptor row counts differ")
        return 0.5 * chemical + 0.5 * physical

    def against_train(self, fingerprints, descriptors):
        return self.kernel(fingerprints, descriptors, self.train_fingerprints,
                           self.train_descriptors)


def protein_features(sequences):
    """AA frequencies (20) followed by ordered dipeptide frequencies (400).

    Each block sums to one, except the all-zero dipeptide block for length-one
    sequences. AA order is ``AMINO_ACIDS``; dipeptide index is 20 * first + second.
    Whitespace/case are normalized. Empty or non-canonical sequences raise;
    ambiguous residues are never silently removed or imputed.
    """
    if isinstance(sequences, str):
        raise ValueError("Supply a sequence of protein strings, not one scalar string")
    sequences = list(sequences)
    result = np.zeros((len(sequences), 420), dtype=float)
    lookup = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    for i, sequence in enumerate(sequences):
        if not isinstance(sequence, str):
            raise ValueError(f"Protein sequence at index {i} is not a string")
        sequence = "".join(sequence.split()).upper()
        if not sequence or set(sequence) - set(lookup):
            raise ValueError(f"Empty or non-canonical protein sequence at index {i}")
        codes = np.asarray([lookup[aa] for aa in sequence])
        result[i, :20] = np.bincount(codes, minlength=20) / len(codes)
        if len(codes) > 1:
            pairs = 20 * codes[:-1] + codes[1:]
            result[i, 20:] = np.bincount(pairs, minlength=400) / (len(codes) - 1)
    return result


def _group_members(groups):
    values = np.asarray(groups, dtype=object)
    if values.ndim != 1:
        raise ValueError("groups must be a one-dimensional array")
    members = {}
    for i, value in enumerate(values):
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise ValueError("Group identities must be strings or integers")
        key = (type(value).__name__, str(value))
        members.setdefault(key, []).append(i)
    return members


def _group_priority(key, seed):
    # Explicit serialization and SHA-256 avoid Python's randomized hash and
    # dependence on input row order. Seed changes deterministic group ordering.
    return hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode("utf-8")).digest()


def grouped_folds(groups, n_splits=5, seed=0):
    """Allocate complete identity/scaffold groups to balanced outcome-free folds.

    Larger groups are placed first, with seeded stable ties, into the currently
    smallest fold by row count. A giant group remains intact; row balance is
    approximate. Group allocation is invariant to permutations of the input.
    """
    members = _group_members(groups)
    if isinstance(n_splits, bool) or not isinstance(n_splits, (int, np.integer)) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least two")
    if len(members) < n_splits:
        raise ValueError("There must be at least n_splits distinct groups")
    order = sorted(members, key=lambda key: (-len(members[key]), _group_priority(key, seed)))
    sizes = np.zeros(n_splits, dtype=int)
    folds = np.empty(sum(map(len, members.values())), dtype=int)
    for key in order:
        fold = int(np.argmin(sizes))
        folds[members[key]] = fold
        sizes[fold] += len(members[key])
    return folds


def nested_group_subsets(groups, budgets, seed=0):
    """Nested whole-group subsets of the supplied training rows.

    Budgets are nondecreasing integer *group counts*, not molecule/label counts
    or fractions; unequal groups can yield unequal row increments. Return a
    list of sorted local row-index arrays, one for each requested budget. Supply
    only an outer/inner training partition, then map returned indices back to
    the original rows. Partition membership is the caller's responsibility;
    outcomes are never accepted by this function.
    """
    members = _group_members(groups)
    budgets = list(budgets)
    if any(isinstance(b, bool) or not isinstance(b, (int, np.integer)) for b in budgets):
        raise ValueError("Budgets must be integer group counts")
    if any(b < 0 or b > len(members) for b in budgets) or budgets != sorted(budgets):
        raise ValueError("Budgets must be nondecreasing and within available group counts")
    order = sorted(members, key=lambda key: _group_priority(key, seed))
    return [np.asarray(sorted(i for key in order[:budget] for i in members[key]), dtype=int)
            for budget in budgets]
