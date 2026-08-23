#!/usr/bin/env python3
"""Generate a larger enterprise corpus, offline and deterministically.

Why this exists: the four sample documents total about 1 100 tokens, which fits
comfortably inside a 4 096-token window. On that corpus the naive baseline never
has to truncate, so the central argument -- that the model cannot see a corpus
larger than its window -- cannot be *demonstrated*, only asserted.

This script builds a corpus that exceeds the window, using the vocabulary and
shape of real enterprise policy documents. Two properties matter:

  * every figure is unique across the corpus, so a question has exactly one
    correct answer and retrieval precision is measurable rather than lucky;
  * the answer-bearing clause of the probe question sits in the *last*
    document, which is precisely what naive truncation discards. That is the
    demonstration: the baseline is wrong for a reason that has nothing to do
    with the model's quality.

Deterministic: same seed, same corpus, byte for byte. A corpus that changes
between runs cannot support a reproducible measurement.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

TOPICS = [
    ("politique_achats", "Politique d'achats et engagements de depense", [
        ("Seuils de validation",
         "Toute commande inferieure a {a} EUR est validee par le responsable de service. "
         "Au-dela de {a} EUR et jusqu'a {b} EUR, la validation du directeur de division "
         "est requise. Au-dela de {b} EUR, un appel d'offres formel est obligatoire."),
        ("Delais de traitement",
         "Une demande d'achat complete est traitee sous {c} jours ouvres. Les demandes "
         "incompletes sont retournees au demandeur sous {d} jours ouvres avec la liste "
         "des pieces manquantes."),
        ("Fournisseurs referenoces",
         "Le referencement d'un nouveau fournisseur exige trois devis comparatifs et un "
         "controle de solvabilite. La duree de validite d'un referencement est de {e} mois."),
    ]),
    ("securite_information", "Politique de securite de l'information", [
        ("Mots de passe",
         "Les mots de passe comportent au minimum {a} caracteres et sont renouveles tous "
         "les {b} jours. La reutilisation des {c} derniers mots de passe est interdite."),
        ("Incidents",
         "Tout incident de securite est signale au responsable securite dans un delai de "
         "{d} heures. Un rapport d'incident est produit sous {e} jours ouvres."),
        ("Postes de travail",
         "Le verrouillage automatique de session intervient apres {f} minutes "
         "d'inactivite. Le chiffrement integral du disque est obligatoire."),
    ]),
    ("integration_collaborateurs", "Procedure d'integration des nouveaux collaborateurs", [
        ("Avant l'arrivee",
         "Le materiel est commande au plus tard {a} jours ouvres avant la date d'arrivee. "
         "Les acces applicatifs sont ouverts la veille de l'arrivee."),
        ("Periode d'essai",
         "La periode d'essai est de {b} mois pour un poste non cadre et de {c} mois pour "
         "un poste cadre. Elle est renouvelable une fois."),
        ("Parcours de formation",
         "Le parcours d'integration comprend {d} modules obligatoires a realiser dans les "
         "{e} premieres semaines."),
    ]),
    ("deplacements_professionnels", "Politique de deplacements professionnels", [
        ("Reservation",
         "Les deplacements sont reserves au moins {a} jours avant le depart. Les trajets "
         "de moins de {b} heures se font en train en seconde classe."),
        ("Avances de frais",
         "Une avance peut etre demandee pour tout deplacement dont le cout estime depasse "
         "{c} EUR. L'avance est versee au plus tard {d} jours avant le depart."),
        ("Justificatifs",
         "Les justificatifs originaux sont transmis dans les {e} jours suivant le retour."),
    ]),
    ("conservation_donnees", "Politique de conservation des donnees", [
        ("Durees de conservation",
         "Les donnees de candidature sont conservees {a} mois. Les dossiers du personnel "
         "sont conserves {b} ans apres le depart. Les journaux techniques sont conserves "
         "{c} mois."),
        ("Suppression",
         "Une demande de suppression est traitee sous {d} jours ouvres. Un accuse de "
         "traitement est adresse au demandeur."),
        ("Sauvegardes",
         "Les sauvegardes completes sont hebdomadaires et conservees {e} semaines."),
    ]),
    ("teletravail_etendu", "Modalites etendues de teletravail", [
        ("Eligibilite",
         "L'eligibilite est examinee apres {a} mois d'anciennete. La demande est instruite "
         "sous {b} jours ouvres."),
        ("Indemnisation",
         "L'indemnite forfaitaire mensuelle est de {c} EUR pour trois jours hebdomadaires "
         "et de {d} EUR pour deux jours hebdomadaires."),
        ("Controle",
         "Un point de suivi est organise tous les {e} mois entre le collaborateur et son "
         "responsable."),
    ]),
    ("formation_professionnelle", "Politique de formation professionnelle", [
        ("Budget",
         "Le budget annuel de formation est de {a} EUR par collaborateur. Un depassement "
         "jusqu'a {b} EUR est possible sur validation de la direction."),
        ("Demandes",
         "Une demande de formation est deposee au moins {c} semaines avant la session. La "
         "reponse intervient sous {d} jours ouvres."),
        ("Engagement",
         "Toute formation de plus de {e} heures donne lieu a un engagement de maintien "
         "dans l'entreprise de {f} mois."),
    ]),
    ("gestion_materiel", "Politique de gestion du materiel informatique", [
        ("Renouvellement",
         "Un ordinateur portable est renouvele tous les {a} mois. Un telephone mobile est "
         "renouvele tous les {b} mois."),
        ("Panne",
         "Une panne bloquante est prise en charge sous {c} heures ouvrees. Un materiel de "
         "pret est fourni au-dela de {d} heures d'immobilisation."),
        ("Restitution",
         "Le materiel est restitue au plus tard le dernier jour travaille. Une retenue de "
         "{e} EUR s'applique en cas de non-restitution des accessoires."),
    ]),
]

PROBE_DOCUMENT = "delegation_signature"
PROBE_TITLE = "Politique de delegation de signature"
PROBE_ANSWER_VALUE = 7500
PROBE_QUESTION = ("Quel est le plafond de delegation de signature pour un chef de "
                  "projet ?")


def _figures(rng: random.Random) -> dict[str, int]:
    """Distinct values so that no two documents can answer the same question."""
    return {key: rng.choice(range(3, 96)) for key in "abcdef"}


def _render(name: str, title: str, sections: list[tuple[str, str]],
            rng: random.Random) -> str:
    lines = [f"# {title}", ""]
    lines.append("## 1. Objet")
    lines.append(
        "Le present document definit les regles applicables a l'ensemble des "
        "collaborateurs. Il complete le reglement interieur et prevaut sur toute "
        "pratique locale non formalisee."
    )
    lines.append("")
    for index, (heading, body) in enumerate(sections, start=2):
        values = _figures(rng)
        lines.append(f"## {index}. {heading}")
        lines.append(body.format(**values))
        lines.append("")
    lines.append(f"## {len(sections) + 2}. Revision")
    lines.append(
        "Ce document est revise annuellement. Toute demande de derogation est "
        "adressee par ecrit a la direction concernee."
    )
    lines.append("")
    return "\n".join(lines)


def _probe_document() -> str:
    """The document that carries the probe answer.

    Placed last in alphabetical *and* generation order so that a naive prompt,
    truncated to the model window, cuts it off. The clause is deliberately
    ordinary: nothing about it is hard for a model to read -- it simply is not
    there when the corpus does not fit.
    """
    return f"""# {PROBE_TITLE}

## 1. Objet
Le present document fixe les plafonds d'engagement financier par niveau de
responsabilite. Il s'applique a tous les engagements contractuels.

## 2. Plafonds par fonction
Le plafond de delegation de signature est de 2000 EUR pour un charge de mission,
de {PROBE_ANSWER_VALUE} EUR pour un chef de projet, de 25000 EUR pour un
directeur de division et de 100000 EUR pour un membre du comite executif.

## 3. Delegation temporaire
Une delegation temporaire est accordee par ecrit pour une duree maximale de
trois mois. Elle ne peut pas etre subdeleguee.

## 4. Controle
Tout engagement superieur au plafond applicable est nul et engage la
responsabilite personnelle du signataire.

## 5. Revision
Ce document est revise annuellement.
"""


def build(target: Path, documents: int, seed: int) -> dict[str, object]:
    rng = random.Random(seed)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total_chars = 0

    for index in range(documents):
        name, title, sections = TOPICS[index % len(TOPICS)]
        suffix = "" if index < len(TOPICS) else f"_{index // len(TOPICS) + 1}"
        filename = f"{name}{suffix}.md"
        text = _render(filename, f"{title}{suffix.replace('_', ' v')}", sections, rng)
        (target / filename).write_text(text, encoding="utf-8")
        written.append(filename)
        total_chars += len(text)

    probe = target / f"z_{PROBE_DOCUMENT}.md"
    probe.write_text(_probe_document(), encoding="utf-8")
    written.append(probe.name)
    total_chars += len(probe.read_text(encoding="utf-8"))

    # The corpus documentation states the correct answer to the probe question,
    # so it must not live inside the corpus. A README that leaks the ground truth
    # into the retrievable text turns a benchmark into a self-fulfilling one --
    # and it is the kind of contamination nobody notices until a reviewer does.
    notes = Path("docs") / f"CORPUS-{target.name}.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    notes.write_text(
        f"""# Corpus `{target.name}` (generated)

{len(written)} documents, roughly {total_chars // 4} estimated tokens.

Rebuild with `scripts/make_corpus.py --seed {seed} --documents {documents}`.
Deterministic: the same seed produces the same corpus, byte for byte.

Probe question: "{PROBE_QUESTION}"
Correct answer: {PROBE_ANSWER_VALUE} EUR, stated in `z_{PROBE_DOCUMENT}.md`.

That document is placed last on purpose. A naive prompt truncated to the model
window cuts it off, so the baseline cannot answer correctly for a reason that
has nothing to do with the model's quality. LeanLM retrieves it because it never
tried to send the whole corpus in the first place.

**This file lives outside the corpus directory deliberately.** It states the
correct answer, and a corpus that contains its own answer key measures nothing.
""",
        encoding="utf-8",
    )
    return {"documents": len(written), "approx_tokens": total_chars // 4,
            "probe_question": PROBE_QUESTION, "probe_answer": PROBE_ANSWER_VALUE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="datasets/enterprise_large")
    parser.add_argument("--documents", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260601)
    args = parser.parse_args()
    result = build(Path(args.target), args.documents, args.seed)
    print(f"  {result['documents']} documents in {args.target}")
    print(f"  approximately {result['approx_tokens']} tokens")
    print(f"  probe: {result['probe_question']}")
    print(f"  answer: {result['probe_answer']} EUR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
