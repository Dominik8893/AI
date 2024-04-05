
import numpy as np

# Cities coordinates
c1_x = np.array([0, 3, 6, 7, 15, 12, 14, 9, 7, 0])
c1_y = np.array([1, 4, 5, 3, 0, 4, 10, 6, 9, 10])

#cities 2
c2_x = np.array([0, 2, 6, 7,  15,   12, 14, 9.5, 7.5, 0.5]);
c2_y = np.array([1, 3, 5, 2.5, -0.5, 3.5, 10, 7.5, 9, 10]);

#cities 3
c3_x = np.array([0, 3, 6, 7,  15,   10, 16, 5, 8, 1.5]);
c3_y = np.array([1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]);

#cities 4
c4_x = np.array([3, 2, 12, 7,  9,  3, 16, 11, 9, 2]);
c4_y = np.array([1, 4, 2, 4.5, 9, 1.5, 11, 8, 10, 7]);


# Population size
Size = 250
ParentSelectionAmount = 0.2
IterationAmount = 50




# Function to calculate distance between 2 points
def DistanceBetweenCities(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to calculate distance for the entire Path
def TourDistance(Path):
    CitiesX = c4_x
    CitiesY = c4_y
    TotalDistance = 0
    for i in range(len(Path)):
        if i == len(Path) - 1:
            TotalDistance += DistanceBetweenCities(CitiesX[Path[i]], CitiesY[Path[i]], CitiesX[Path[0]], CitiesY[Path[0]])
        else:
            TotalDistance += DistanceBetweenCities(CitiesX[Path[i]], CitiesY[Path[i]], CitiesX[Path[i+1]], CitiesY[Path[i+1]])
    return TotalDistance


def CycleCrossover(Parent1, Parent2):
    Offspring1 = [None] * len(Parent1)
    Offspring2 = [None] * len(Parent2)

    # Choose a random starting point
    start_index = np.random.randint(0, len(Parent1))
    index = start_index

    # Perform cycle crossover
    while True:
        # Copy the selected elements from parent 1 to offspring 1
        Offspring1[index] = Parent1[index]

        # Find the element from parent 2 corresponding to the current index
        element = Parent2[index]
        index = np.where(Parent1 == element)[0][0]

        # Check if the cycle is complete
        if index == start_index:
            break

    # Fill in the remaining elements for offspring 2
    for i in range(len(Offspring2)):
        if Offspring1[i] is None:
            Offspring1[i] = Parent2[i]
            Offspring2[i] = Parent1[i]
        else:
            Offspring2[i] = Parent1[i]

    return Offspring1, Offspring2

def Mutate(ElementToBeMutated):
    #select Elements to swap
    FirstSwap = np.random.randint(0,9)
    SecondSwap = np.random.randint(0, 9)

    #perform Swap
    Temp = np.copy(ElementToBeMutated)
    ElementToBeMutated[FirstSwap] = Temp[SecondSwap]
    ElementToBeMutated[SecondSwap] = Temp[FirstSwap]

    return ElementToBeMutated





# Initialize population
Population = np.zeros((Size, 10),dtype = int )
Distances = np.zeros(Size)

# Generate random population
for i in range(Size):
    random_arr = np.random.permutation(10)
    Population[i, :] = random_arr
    Distances[i] = TourDistance(Population[i])

for j in range(IterationAmount):
    SelectedParents = np.zeros((round(Size*ParentSelectionAmount), 10),dtype = int )
    for i in range(round(Size*ParentSelectionAmount)):
        SelectedParents[i] = Population[i]

    Children = np.zeros((round(Size*ParentSelectionAmount), 10),dtype = int )
    for i in range(round((Size*ParentSelectionAmount)/2)):

        Parent1 = SelectedParents[np.random.choice(SelectedParents.shape[0])]
        Parent2 = SelectedParents[np.random.choice(SelectedParents.shape[0])]

        Children[i], Children[i + round((Size*ParentSelectionAmount)/2)] = CycleCrossover(Parent1, Parent2)

    #mutate children
    for i in range(round(Size*ParentSelectionAmount)):
        Children[i] = Mutate(Children[i])


    Population = np.concatenate((Population, Children), axis=0)

    #sorting the population by best distance
    Population = sorted(Population, key=TourDistance)

    #cutting off excess population
    Population = Population[:Size]

    print("Best Paren in " + str(j) + "  " + str(TourDistance(Population[0])))
