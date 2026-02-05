# Physics-Informed Neural Networks (PINNs)

Implémentation de réseaux de neurones informés par la physique pour résoudre des équations aux dérivées partielles. L'idée est d' intégrer directement les équations dans la fonction de perte. Le réseau apprend à satisfaire les lois physiques.

## Contenu du projet

### Équation de la chaleur (heat.py)

Résolution de l'équation de diffusion en 2D+temps. La condition initiale est un produit de sinus, et j'ai mis des conditions de Dirichlet homogène aux bords. Le réseau a appris comment la température se diffusait au fil du temps. La visualisation interactive avec slider permet de voir comment le profil change.
![pret_gif_chaleur](https://github.com/user-attachments/assets/0799af4e-393c-41b7-b760-a5c3b3401be4)

### Problème inverse (heat_inverse.py)

Au lieu de résoudre avec un coefficient de diffusion fixe, on l'apprend directement à partir de mesures virtuelles placées dans le domaine. C'est utile en pratique : dans beaucoup de situations réelles, on connaît mal les paramètres physiques (conductivité thermique, viscosité, perméabilité d'un matériau). Ici le réseau identifie le paramètre inconnu en minimisant l'écart entre ses prédictions et les données des capteurs, il fait alors partie des paramètres du modèle.
![pret_gif_inverse](https://github.com/user-attachments/assets/e305d26f-5e6a-4eb2-833c-1452b87aca33)
### Équations de Burgers (burger.py)

Système non-linéaire avec advection et diffusion. C'est plus intéressant que la chaleur parce qu'il y a des phénomènes d'onde. J'ai pu vérifier que les solutions convergent correctement quand on les compare aux solutions analytiques.

![pret_gif_uburger](https://github.com/user-attachments/assets/0d0757bf-7fd6-4131-85b7-81fede3df652)
![pret_gif_burgerv](https://github.com/user-attachments/assets/10f3dde8-d6fc-4e29-ba29-72d7d4066f64)

### Équations de Navier-Stokes (navier.py)

Le plus complexe : flux autour d'un cylindre en 2D. Il faut gérer trois inconnues (u, v, p), trois équations (momentum x, momentum y, continuité), et des conditions de no-slip sur l'obstacle. J'utilise une optimisation en deux phases : d'abord Adam pour la convergence globale, puis L-BFGS pour affiner. Une capacité de calcul plus grande aurait pu permettre d'observer les célèbres allées de tourbillons de Karman.



## Comment ça marche

Le réseau prend en entrée les coordonnées spatiales et temporelles (x, y, t) et retourne la solution (u, v, p, etc.). 

La perte à minimiser a plusieurs composantes :
- La résidu de l'EDP elle-même (les dérivées partielles calculées via autograd)
- Les conditions initiales
- Les conditions au bord
- Les éventuelles contraintes (obstacles, no-slip)

Les dérivées partielles sont calculées avec `torch.autograd.grad()` en utilisant `create_graph=True` pour les dérivées secondes. C'est coûteux, donc je sample les points avec Latin Hypercube pour couvrir le domaine de manière uniforme tout en restant aléatoire.

J'ai utilisé principalement Adam pour l'optimisation, cependant l'équation de Navier-Stockes étant complexe, j'ai affiner le modèle grâce à l'optimiseur L-FBGS (ordre 2).


## Dépendances

- PyTorch (différenciation automatique)
- SciPy (Latin Hypercube Sampling)
- Matplotlib (visualisation)
- NumPy

Pour GPU : CUDA ou MPS (M1/M2/M3 Apple Silicon)

## Ce que j'ai appris

L'approche PINN m'a montré comment combiner deep learning et physique numérique. L'enjeu principal est la balance entre les termes de perte : si on pèse trop les conditions au bord, le réseau les satisfait parfaitement mais ignore l'EDP.

Le calcul des dérivées par autograd peut être coûteux sur grandes grilles, donc l'optimisation du sampling et du nombre de points est crucial.
