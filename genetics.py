# Imports
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling


# ---------Create nodes and conditional probability distribution ------

def get_probs_gene_ancestor(varName):
    """
    Builds a conditional probability distribution (TabularCPD)
    using a priori probabilities, for a variable gene with no
    parents

    Parameter
    ----------
    varName : name of the variable (String)
    """
    cpd = TabularCPD (
        variable = varName,  # Gene | Gene_Father | Gene_Mother
        variable_card = 3,  # P(Gene)(2 | 1 | 0)
        values = [
            [0.01], [0.03], [0.96]
        ],
        state_names = {
            varName: ["2", "1", "0"]
        }
    )
    #print(cpd)
    return cpd

#get_probs_gene_ancestor("Gene")     # Test de l'affichage du TabularCPD

# +------------+------+
# | P(Gene)(2) | 0.01 |
# +------------+------+
# | P(Gene)(1) | 0.03 |
# +------------+------+
# | P(Gene)(0) | 0.96 |
# +------------+------+

def get_probs_trait(varName,evidenceName):
    """
    Builds a conditional probability distribution (TabularCPD)
    for traits given the number of genes

    Parameters
    ----------
    varName : name of the traits variable (String)
    evidenceName: name of the evidence gene variable (String)
    """
    cpd = TabularCPD (
        variable = varName,
        variable_card = 2, 
        values = [
            [0.65, 0.56, 0.01], # Trait = vrai
            [0.35, 0.44, 0.99]  # Trait = faux
        ],
        evidence = [evidenceName],
        evidence_card = [3],    # 0, 1, 2
        state_names = {
            varName: ["vrai", "faux"],
            evidenceName: ["2", "1", "0"]
        },
    )
    #print(cpd)
    return cpd

get_probs_trait("Trait", "Gene")    # Test de l'affichage du TabularCPD

# +-----------------------+---------+---------+---------+
# | Gene                  | Gene(2) | Gene(1) | Gene(0) |
# +-----------------------+---------+---------+---------+
# | P(Trait | Gene)(vrai) | 0.65    | 0.56    | 0.01    |
# +-----------------------+---------+---------+---------+
# | P(Trait | Gene)(faux) | 0.35    | 0.44    | 0.99    |
# +-----------------------+---------+---------+---------+

# constant defining mutation probability of a gene
prob_mutation = 0.01    # probabilité que le gène devienne muté / faux muté après transmission

def get_probs_heredity1(geneParent):
    """
    Computes probability of inheriting 1 gene from
    a given parent: 
    P(Gene_{inherited chromosome}|Gene_{parent})

    Parameter
    ----------
    geneParent: number of genes (0, 1 or 2) of the
    parent (father or mother)
    """
    # 3 valeurs possibles pour geneParent (0, 1, 2) :
    if geneParent == 0:    # Le parent n'a pas de gène 
        return prob_mutation
    elif geneParent == 1: 
        return 0.5 + 0.5 * prob_mutation    # Probabilité que le gène transmis soit le gène muté + que le gène transmis soit le gène faux muté
    elif geneParent == 2:
        return 1
    else:
        raise ValueError("La parent peut transmettre 0, 1 ou 2 gènes.")

# Affichage des 3 probabilités
#print("P(G enfant | 0) = " + str(get_probs_heredity1(0)))
#print("P(G enfant | 1) = " + str(get_probs_heredity1(1)))
#print("P(G enfant | 2) = " + str(get_probs_heredity1(2)))

def get_probs_gene(varNameChild,evidenceNameFather,evidenceNameMother):
    """
    Builds a conditional probability distribution (TabularCPD)
    for the number of genes of a child given the number of genes
    of each of the parents

    Parameters
    ----------
    varNameChild : name of the traits variable (String)
    evidenceName: name of the evidence gene variable (String)
    """
    
    # Récupération des probabilités des 3 valeurs possibles avec get_probs_heredity(0 | 1 | 2)
    p_heredity_0 = get_probs_heredity1(0)
    p_heredity_1 = get_probs_heredity1(1)
    p_heredity_2 = get_probs_heredity1(2)

    # Stockage des valeurs dans un dictionnaire
    probs = {
        0: p_heredity_0,
        1: p_heredity_1,
        2: p_heredity_2
    }

    # Déclaration de 3 listes vides
    row_gene_child_2 = []
    row_gene_child_1 = []
    row_gene_child_0 = []

    # Calcul des combinaisons
    for father in [0, 1, 2]:
        pf = probs[father]  # Récupération de la probabilité du père
        for mother in [0, 1, 2]:
            pm = probs[mother]  # Récupération de la probabilité de la mère
            # Calcul des 3 probabilités de Gene_Child = 0 | 1 | 2
            p0 = (1 - pf) * (1 - pm)
            row_gene_child_0.append(p0)
            p1 = (pf * (1 - pm)) + (pm * (1 - pf))
            row_gene_child_1.append(p1)
            p2 = pf * pm
            row_gene_child_2.append(p2)


    cpd = TabularCPD (
        variable = varNameChild,
        variable_card = 3, # "Gene_Child=2", "Gene_Child=1", "Gene_Child=0"
        values = [
            row_gene_child_2, # Gene_Child=2
            row_gene_child_1, # Gene_Child=1
            row_gene_child_0  # Gene_Child=0
        ],
        evidence = [evidenceNameFather, evidenceNameMother],
        evidence_card = [3, 3],
        state_names = {
            evidenceNameFather: ["2", "1", "0"],
            evidenceNameMother: ["2", "1", "0"],
            varNameChild: ["2", "1", "0"]
        },
    )
    #print(cpd)
    return cpd

# +---------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
# | Gene_Father   | Gene_Father(2) | Gene_Father(2) | Gene_Father(2) | Gene_Father(1) | Gene_Father(1) | Gene_Father(1) | Gene_Father(0) | Gene_Father(0) | Gene_Father(0) |
# +---------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
# | Gene_Mother   | Gene_Mother(2) | Gene_Mother(1) | Gene_Mother(0) | Gene_Mother(2) | Gene_Mother(1) | Gene_Mother(0) | Gene_Mother(2) | Gene_Mother(1) | Gene_Mother(0) |
# +---------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
# | Gene_Child(2) | 0.0001         | 0.00505        | 0.01           | 0.00505        | 0.255025       | 0.505          | 0.01           | 0.505          | 1.0            |
# +---------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
# | Gene_Child(1) | 0.0198         | 0.5049         | 0.99           | 0.5049         | 0.49995        | 0.495          | 0.99           | 0.495          | 0.0            |
# +---------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
# | Gene_Child(0) | 0.9801         | 0.49005        | 0.0            | 0.49005        | 0.245025       | 0.0            | 0.0            | 0.0            | 0.0            |
# +---------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+


#--------------------Create a Bayesian Network for family n°1 --------------------------------

# Création du modèle 
model1 = DiscreteBayesianNetwork([

    # Définition pour les gènes
    ("Gene_Leto", "Gene_Paul"),
    ("Gene_Leto", "Gene_Alia"),
    ("Gene_Jessica", "Gene_Paul"),
    ("Gene_Jessica", "Gene_Alia"),

    # Définition pour les traits
    ("Gene_Leto", "Trait_Leto"),
    ("Gene_Jessica", "Trait_Jessica"),
    ("Gene_Paul", "Trait_Paul"),
    ("Gene_Alia", "Trait_Alia")
])

# Création des TabularCPD pour les probabilités sur les gènes
cpd_genes_leto = get_probs_gene_ancestor("Gene_Leto")
cpd_genes_jessica = get_probs_gene_ancestor("Gene_Jessica")
cpd_genes_paul = get_probs_gene("Gene_Paul", "Gene_Leto", "Gene_Jessica")
cpd_genes_alia = get_probs_gene("Gene_Alia", "Gene_Leto", "Gene_Jessica")

# Création des TabularCPD pour les probabilités sur les traits
cpd_trait_leto = get_probs_trait("Trait_Leto", "Gene_Leto")
cpd_trait_jessica = get_probs_trait("Trait_Jessica", "Gene_Jessica")
cpd_trait_paul = get_probs_trait("Trait_Paul", "Gene_Paul")
cpd_trait_alia = get_probs_trait("Trait_Alia", "Gene_Alia")

# Ajout des TabularCPD au modèle
model1.add_cpds(

    # TabularCPD sur les gènes
    cpd_genes_leto,
    cpd_genes_jessica,
    cpd_genes_paul,
    cpd_genes_alia,

    # TabularCPD sur les traits
    cpd_trait_leto,
    cpd_trait_jessica,
    cpd_trait_paul, 
    cpd_trait_alia
)  

# Renvoie "True" si en aditionnant chaque colonne, on obtient 1, ce qui permet de vérifier que les probabilités sont bien calculées.
print(model1.check_model()) 

# Vérification du modèle graphiquement
viz = model1.to_graphviz()
viz.draw('family1.png', prog='dot')

#--------------------Inference for family n°1 --------------------------------

# Exact inference 
infer = VariableElimination(model1)

# Calculate predictions based on the evidence provided by the Trait variables
T = {
    "Trait_Leto": "faux",
    "Trait_Jessica": "vrai",
    #"Trait_Paul": "faux",
    "Trait_Alia": "vrai"
}

result = infer.query(variables = ["Gene_Paul"], evidence = T)
print(result)

# Calculate predictions based on the evidence provided by the Trait variables and knowing that Jessica and Alia have resp. 1 and 2 genes
T = {
    "Trait_Leto": "faux",
    "Trait_Jessica": "vrai",
    #"Trait_Paul": "faux",
    "Trait_Alia": "vrai",
    "Gene_Jessica": "1",
    "Gene_Alia": "2"
}

result = infer.query(variables = ["Gene_Paul"], evidence = T)
print(result)

# Approximate inference
inference = BayesianModelSampling(model1)

# This method gets samples, until 5000 samples with "Trait_Paul" fixed to "vrai" are collected. 
# Samples which do not satisfy this condition are rejected.
evidence = [
    State(var='Trait_Leto', state='faux'),
    State(var='Trait_Jessica', state='vrai'),
    State(var='Trait_Alia', state='vrai'),
    State(var='Gene_Jessica', state='1'),
    State(var='Gene_Alia', state='2')
]
samples = inference.rejection_sample(evidence=evidence, size=5000)
series_RV = samples['Gene_Paul'].value_counts()
#print(series_RV)

# Affichage des probabilités
print("P(G Paul = 2 | T) = % .4f" %(series_RV["2"]/sum(series_RV)))
print("P(G Paul = 1 | T) = % .4f" %(series_RV["1"]/sum(series_RV)))
print("P(G Paul = 0 | T) = % .4f" %(series_RV["0"]/sum(series_RV)))

#--------------------Create a Bayesian Network for family n°2 --------------------------------

# Liste des personnes formant la famille n°2
lstP = ['Charles', 'Diana', 'Michael', 'Carole', 'Harry', 'Meghan', 'William', 'Katherine', 'Philippa',
        'Archie', 'Liliet', 'George', 'Charlotte', 'Lvrais']

# Structure de données pour y indiquer les liens de parentés
royal_family_dict = {
    # Première lignée
    "Charles": {"parents": []},
    "Diana": {"parents": []},
    "Michael": {"parents": []},
    "Carole": {"parents": []},

    # Seconde lignée
    "Harry": {"parents": ["Charles", "Diana"]},
    "Meghan": {"parents": []},
    "William": {"parents": ["Charles", "Diana"]},
    "Katherine": {"parents": ["Michael", "Carole"]},
    "Philippa": {"parents": ["Michael", "Carole"]},

    # Troisième lignée
    "Archie": {"parents": ["Harry", "Meghan"]},
    "Liliet": {"parents": ["Harry", "Meghan"]},
    "George": {"parents": ["William", "Katherine"]},
    "Charlotte": {"parents": ["William", "Katherine"]},
    "Lvrais": {"parents": ["William", "Katherine"]},
}

# Fonction qui retourne le DiscreteBayesianNetwork crée avec toutes les règles
def get_royal_family_model(family_dict):
    edges = []

    for person, values in royal_family_dict.items():
        parents = values["parents"]

        # Ajout des règles sur les liens de parentés
        for parent in parents:
            edges.append((f"Gene_{parent}", f"Gene_{person}"))

        # Ajout des liens entre le nombre de gènes et les traits
        edges.append((f"Gene_{person}", f"T_{person}"))

    # Création du modèle
    model = DiscreteBayesianNetwork(edges)
    return model

model2 = get_royal_family_model(royal_family_dict)

def get_family_members_cpds(family_dict):
    cpds = []

    for person, values in royal_family_dict.items():
        parents = values["parents"]
        gene = f"Gene_{person}"
        trait = f"T_{person}"

        # Cas de la première lignée (aucune donnée sur les parents)
        if len(parents) == 0:
            cpds.append(get_probs_gene_ancestor(gene))

        # Cas où il s'agit d'un fils et on a des données sur les parents
        elif len(parents) == 2:
            father, mother = parents
            cpds.append(get_probs_gene(gene, f"Gene_{father}", f"Gene_{mother}"))

        # Définition du CPD pour le trait
        cpds.append(get_probs_trait(trait, gene))

    return cpds

# Ajout des TabularCPD
cpds = get_family_members_cpds(royal_family_dict)
model2.add_cpds(*cpds)

# Renvoie "True" si en aditionnant chaque colonne, on obtient 1, ce qui permet de vérifier que les probabilités sont bien calculées.
print(model2.check_model())

# Vérification du modèle graphiquement
viz = model2.to_graphviz()
viz.draw('family2.png', prog='dot')

#--------------------Inference for family n°2 --------------------------------

# Exact inference 
infer = VariableElimination(model2)
dicT = {'T_Charles': 'vrai', 'T_Diana': 'faux', 'T_Michael': 'faux', 'T_Carole': 'faux',
        'T_Harry': 'faux', 'T_Meghan': 'faux', 'T_William': 'faux', 'T_Katherine': 'faux', 'T_Philippa': 'vrai',
        'T_Archie': 'vrai', 'T_Liliet': 'faux', 'T_George': 'faux', 'T_Charlotte': 'faux', 'T_Lvrais': 'vrai'}

# Calculate predictions based on the evidence provided by the Trait variables
george_result1 = infer.query(variables = ["Gene_George"], evidence = dicT)
print(george_result1)

liliet_result1 = infer.query(variables = ["Gene_Liliet"], evidence = dicT)
print(liliet_result1)

# Calculate predictions based on the evidence provided by the Trait variables and knowing that Meghan has no gene
dicT = {'T_Charles': 'vrai', 'T_Diana': 'faux', 'T_Michael': 'faux', 'T_Carole': 'faux',
        'T_Harry': 'faux', 'T_Meghan': 'faux', 'T_William': 'faux', 'T_Katherine': 'faux', 'T_Philippa': 'vrai',
        'T_Archie': 'vrai', 'T_Liliet': 'faux', 'T_George': 'faux', 'T_Charlotte': 'faux', 'T_Lvrais': 'vrai',
        # Ajout de cette ligne pour préciser le problème :
        'Gene_Katherine': '0'}

george_result2 = infer.query(variables = ["Gene_George"], evidence = dicT)
print(george_result2)

liliet_result2 = infer.query(variables = ["Gene_Liliet"], evidence = dicT)
print(liliet_result2)