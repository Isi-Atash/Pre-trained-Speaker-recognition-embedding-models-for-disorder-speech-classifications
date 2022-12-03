import pandas as pd

xvector_hun = pd.read_csv('Xvector_embedding_hun.csv')
xvector_dutch = pd.read_csv('Xvector_embedding_dutch.csv')
ecapa_hun = pd.read_csv('Ecapa_embedding_hun.csv')
ecapa_dutch = pd.read_csv('Ecapa_embedding_dutch.csv')

c = xvector_hun[(xvector_hun.label == 'OD') | (xvector_hun.label == 'HC')]
xvector_dutch = xvector_dutch[(xvector_dutch.label == 'OD') | (xvector_dutch.label == 'HC')]
ecapa_hun = ecapa_hun[(ecapa_hun.label == 'OD') | (ecapa_hun.label == 'HC')]
ecapa_dutch = ecapa_dutch[(ecapa_dutch.label == 'OD') | (ecapa_dutch.label == 'HC')]