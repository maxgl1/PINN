# Physics-Informed Neural Networks (PINNs)

Implémentation de réseaux de neurones informés par la physique pour résoudre des équations aux dérivées partielles. L'idée est d' intégrer directement les équations dans la fonction de perte. Le réseau apprend à satisfaire les lois physiques.

## Contenu du projet

### Équation de la chaleur (heat.py)

Résolution de l'équation de diffusion en 2D+temps. La condition initiale est un produit de sinus, et j'ai mis des conditions de Dirichlet homogène aux bords. Le réseau a appris comment la température se diffusait au fil du temps. La visualisation interactive avec slider permet de voir comment le profil change.

$$ \frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$


<p align="center"> <b>Visualisation de la diffusion thermique</b>


<video src="https://github.com/user-attachments/assets/9bb33495-bfa5-40fa-9fd5-5f47508f2173" width="70%" autoplay loop muted></video> </p>





### Problème inverse (heat_inverse.py)

Au lieu de résoudre avec un coefficient de diffusion fixe, on l'apprend directement à partir de mesures virtuelles placées dans le domaine. C'est utile en pratique : dans beaucoup de situations réelles, on connaît mal les paramètres physiques (conductivité thermique, viscosité, perméabilité d'un matériau). Ici le réseau identifie le paramètre inconnu en minimisant l'écart entre ses prédictions et les données des capteurs, il fait alors parti des paramètres du modèle. Ici, seulement 4 capteurs ont suffi pour estimer la valeur réelle.

$$
\begin{cases} 
\alpha_{réel} = 0.5 \\ 
\alpha_{pinn} = 0.52 
\end{cases} 
\implies \epsilon_{relatif} = 4.0 \times 10^{-2}
$$
<p align="center"> <b>Identification du paramètre inconnu, capteurs positionné sur les croix</b>


<video src="https://github.com/user-attachments/assets/3475a8cc-8fb0-40ef-94b8-5655d0959caf" width="70%" autoplay loop muted></video> </p>





### Équations de Burgers (burger.py)

Système non-linéaire avec advection et diffusion. C'est plus intéressant que la chaleur parce qu'il y a des phénomènes d'onde. J'ai pu vérifier que les solutions convergent correctement quand on les compare aux solutions analytiques.

$$
\begin{cases}
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
\end{cases}
$$

<table width="100%"> <tr> <td align="center" width="50%"> <b>Profil des vitesses de u au cours du temps</b>


<video src="https://github.com/user-attachments/assets/fbf6ad1f-eba1-4838-a437-2bb3f802ed80" width="100%" autoplay loop muted></video> </td> <td align="center" width="50%"> <b>Profil des vitesses de v au cours du temps</b>


<video src="https://github.com/user-attachments/assets/f233f09c-94c2-4dec-8227-ce85e66d8e03" width="100%" autoplay loop muted></video> </td> </tr> </table>









### Équations de Navier-Stokes (navier.py)

Le plus complexe : flux autour d'un cylindre en 2D. Il faut gérer trois inconnues (u, v, p), trois équations (momentum x, momentum y, continuité), et des conditions de no-slip sur l'obstacle. J'utilise une optimisation en deux phases : d'abord Adam pour la convergence globale, puis L-BFGS pour affiner. Une capacité de calcul plus grande aurait pu permettre d'observer les célèbres allées de tourbillons de Karman.

$$
\begin{cases} 
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0 \\ 
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) \\ 
\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right) 
\end{cases}
$$

<table width="100%"> <tr> <td align="center" width="50%"> <b>Profil des vitesses (u) au cours du temps</b>


<video src="https://github.com/user-attachments/assets/f18529b4-aab4-4022-a297-569f1155cca1" width="100%" autoplay loop muted></video> </td> <td align="center" width="50%"> <b>Profil des vitesses (v) au cours du temps</b>


<video src="https://github.com/user-attachments/assets/42068656-0682-491f-a8f7-19ecb1fbb581" width="100%" autoplay loop muted></video> </td> </tr> </table>











## Comment ça marche

Le réseau prend en entrée les coordonnées spatiales et temporelles (x, y, t) et retourne la solution (u, v, p, etc.). 

La perte à minimiser a plusieurs composantes :
- La résidu de l'EDP elle-même (les dérivées partielles calculées via autograd)
- Les conditions initiales
- Les conditions au bord
- Les éventuelles contraintes (obstacles, no-slip)

$$ \mathcal{L} = \mathcal{L}_{init} + \mathcal{L}_{residu} + \mathcal{L}_{bord} + \mathcal{L}_{obstacles}$$

Les dérivées partielles sont calculées avec `torch.autograd.grad()` en utilisant `create_graph=True` pour les dérivées secondes. C'est coûteux, donc je sample les points avec Latin Hypercube pour couvrir le domaine de manière uniforme tout en restant aléatoire.

J'ai utilisé principalement Adam pour l'optimisation, cependant l'équation de Navier-Stokes étant complexe, j'ai affiné le modèle grâce à l'optimiseur L-BFGS (ordre 2).


## Dépendances

- PyTorch (différenciation automatique)
- SciPy (Latin Hypercube Sampling)
- Matplotlib (visualisation)
- NumPy

Pour GPU : CUDA ou MPS (M1/M2/M3 Apple Silicon)
