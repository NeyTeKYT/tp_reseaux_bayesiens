# Imports
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

# ------------------------------------Create Nodes------------------------------------

# meteo node has no parents
probs_meteo = TabularCPD(
    variable="Météo",
    variable_card=3, 
    values=[[0.7], # ensoleillé
        [0.2], # pluie
        [0.1]  # neige
        ],
    state_names={"Météo" : ["ensoleillé", "pluie", "neige"]}
)

# accident node is conditional on meteo
probs_accident = TabularCPD(
    variable="Accident",
    variable_card=2,
    values=[[0.1, 0.2, 0.4], # oui
            [0.9, 0.8, 0.6]], # non
    evidence=["Météo"],
    evidence_card=[3],
    state_names={"Accident": ["oui","non"],
                 "Météo" : ["ensoleillé", "pluie", "neige"]}
)
#  +----------------+--------------+-------------+-------------+
#  | Météo          |  ensoleillé  |     pluie   |    neige    |
#  +----------------+--------------+-------------+-------------+
#  | Accident=oui   |      0 1     |     0.2     |     0.4     |
#  +----------------+--------------+-------------+-------------+
#  | Accident=non   |      0.9     |     0.8     |     0.6     |
#  +----------------+--------------+-------------+-------------+


# embouteillage node is conditional on accident and meteo
probs_embouteillage = TabularCPD(
    variable="Embouteillage",
    variable_card=2,
    values=[[0.6, 0.8, 0.9, 0.2, 0.5, 0.7], # oui
            [0.4, 0.2, 0.1, 0.8, 0.5, 0.3]], # non
    evidence=["Accident","Météo"],
    evidence_card=[2,3],
    state_names={"Embouteillage": ["oui","non"],
                 "Accident" : ["oui","non"],
                 "Météo" : ["ensoleillé", "pluie", "neige"]}
)
#  +-------------------+--------------+-------------+-------------+--------------+-------------+-------------+
#  | Accident          |     oui      |     oui     |     oui     |      non     |     non     |     non     |
#  +-------------------+--------------+-------------+-------------+--------------+-------------+-------------+
#  | Météo             |  ensoleillé  |     pluie   |    neige    |  ensoleillé  |     pluie   |    neige    |
#  +-------------------+--------------+-------------+-------------+--------------+-------------+-------------+
#  | Embouteillage=oui |     0 6      |     0.8     |     0.9     |      0 2     |     0.5     |     0.7     |
#  +-------------------+--------------+-------------+-------------+--------------+-------------+-------------+
#  | Embouteillage=non |     0.4      |     0.2     |     0.1     |      0.8     |     0.5     |     0.3     |
#  +-------------------+--------------+-------------+-------------+--------------+-------------+-------------+



# rendez-vous node is conditional on embouteillage
probs_RV = TabularCPD(
    variable="Rendez-vous",
    variable_card=2,
    values=[[0.4, 0.9], # maintenu
            [0.6, 0.1]], # annulé
    evidence=["Embouteillage"],
    evidence_card=[2],
    state_names={"Rendez-vous": ["maintenu","annulé"],
                 "Embouteillage": ["oui","non"]}
)
#  +----------------------+-----------+-----------+
#  | Embouteillage        |    oui    |   non     |
#  +----------------------+-----------+-----------+
#  | Rendez-vous=maintenu |    0.4    |   0.9     |
#  +----------------------+-----------+-----------+
#  | Rendez-vous=annulé   |    0.6    |   0.1     |
#  +----------------------+-----------+-----------+


# ------------------------------------Create a Bayesian Network and add states------------------------------------

model = DiscreteBayesianNetwork(
    [('Météo', 'Accident'),
    ('Météo', 'Embouteillage'),
    ('Accident', 'Embouteillage'),
    ('Embouteillage', 'Rendez-vous')])

model.add_cpds(probs_meteo, probs_accident, probs_embouteillage, probs_RV)
model.check_model()


viz = model.to_graphviz()
viz.draw('rv.png', prog='dot')
# NB: generate a TypError for graphviz

# ------------------------------------Exact inference------------------------------------

from pgmpy.inference import VariableElimination

# Calculate probability for a given observation [météo=pluie, accident=oui, embouteillage=non, rv=annulé]
infer = VariableElimination(model)
g_dist = infer.query(variables=["Météo", "Accident", "Embouteillage", "Rendez-vous"])
g_dist.reduce([('Météo', 'pluie'), ("Accident", "oui"), ("Embouteillage", "non"), ("Rendez-vous", "annulé")])
print("P(météo=pluie, accident=oui, embouteillage=non, rv=annulé)="+'{:.5f}'.format(g_dist.get_value()))

# Calculate predictions based on the evidence that an accident happened
print(infer.query(['Rendez-vous'], evidence={'Accident': 'oui'}))


# ------------------------------------Approximate inference with Sampling------------------------------------

from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelSampling
import pandas as pd

# Objective: Compute distribution of Appointment given that train is delayed


# 1st method: get 10000 samples, then discard samples where no accident happens
inference = BayesianModelSampling(model)
data = inference.forward_sample(size=10000)
print(data.head())

samples = pd.DataFrame(columns=data.columns)
for index, row in data.iterrows():
    # If, in this sample, the variable of accident has the value "oui", save the sample.
    # Since we are interested in the probability distribution of RV given that an accident happens,
    # we discard the samples where no accident occurs.
    if row["Accident"] == "oui":
        samples.loc[len(samples)] = row
print(samples.head())

print(samples["Rendez-vous"].value_counts()["maintenu"])
# Count how many times each value of the variable appeared
print("P(Rendez-vous=maintenu|Accident=oui)=% .4f" %(float(samples["Rendez-vous"].value_counts()["maintenu"])/len(samples)))


# 2nd method : get samples, until 5000 samples with accident fixed to 'oui' are collected. Samples
# which do not satisfy this condition are rejected
evidence = [State(var='Accident', state='oui')]
samples = inference.rejection_sample(evidence=evidence, size=5000)
series_RV = samples['Rendez-vous'].value_counts()
print("P(Rendez-vous=maintenu|Accident=oui)=% .4f" %(series_RV["maintenu"]/sum(series_RV)))
