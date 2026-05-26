import json
import numpy as np
from sys import argv, exit
from scipy.sparse import csc_array, lil_matrix
from abc import ABC, abstractmethod
from scipy.sparse.linalg import spsolve
from matplotlib.pyplot import spy, show
import pyvista as pv

#np.set_printoptions(precision=4, suppress=True)

from Plate20 import (Klocal, Flocal, Blocalbending, Dmatbending, Blocalmembrane, Dmatmembrane)

class mesh_plate(ABC):
    def __init__(self, xnodes: np.array, ynodes: np.array, num_dofs_per_node: int = 5):
        self.xnodes = xnodes
        self.ynodes = ynodes
        self.num_dofs_per_node = num_dofs_per_node
    
    def getNumNodes(self) -> int:
        return self.xnodes.size * self.ynodes.size
    
    def getNumNodes_x(self) -> int:
        return self.xnodes.size
    
    def getNumNodes_y(self) -> int:
        return self.ynodes.size
    
    def getNumElements(self) -> int:
        return (self.xnodes.size - 1) * (self.ynodes.size - 1)
    
    def getNumElements_x(self) -> int:
        return self.xnodes.size - 1
    
    def getNumElements_y(self) -> int:
        return self.ynodes.size - 1
    
    def getNumDofs(self) -> int:
        return self.getNumNodes() * self.num_dofs_per_node
    
    def getNumNodesPerElement(self) -> int:
        return 4
    
    def getNumDofsPerElement(self) -> int:
        return self.getNumNodesPerElement()*self.num_dofs_per_node
    
    @abstractmethod
    def getCoordinatesOfNode(self, nodeNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getDofsOfNode(self, nodeNum: int) -> np.array:
        pass
        
    @abstractmethod
    def getNodesOfElement(self, elementNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getCoordinatesOfNodesOfElement(self, elementNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getDofsOfElement(self, elementNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getSizeOfElement(self, elementNum: int) -> np.array:
        pass

class mesh_plate_enumx(mesh_plate):
    def __init__(self, xnodes: np.array, ynodes: np.array, num_dofs_per_node: int = 5):
        super().__init__(xnodes, ynodes, num_dofs_per_node)
    
    def getCoordinatesOfNode(self, nodeNum: int) -> np.array:
        assert nodeNum <= self.getNumNodes() - 1, f"Node number {nodeNum} exceeds the maximum number of nodes, which is {self.getNumNodes()-1}"
        numNodesBeforeThisNode = nodeNum
        numRowsBeforeThisNode = numNodesBeforeThisNode % self.getNumNodes_x()
        numColsBeforeThisNode = numNodesBeforeThisNode // self.getNumNodes_x()
        xCoordOfThisNode = self.xnodes[numRowsBeforeThisNode]
        yCoordOfThisNode = self.ynodes[numColsBeforeThisNode]
        return np.array([xCoordOfThisNode,yCoordOfThisNode])
    
    def getDofsOfNode(self, nodeNum: int) -> np.array:
        assert nodeNum <= self.getNumNodes() - 1, f"Node number {nodeNum} exceeds the maximum number of nodes, which is {self.getNumNodes()-1}"
        numDofsBeforeThisNode = nodeNum * self.num_dofs_per_node
        dofsOfThisNode = numDofsBeforeThisNode + np.arange(self.num_dofs_per_node)
        return dofsOfThisNode

    def getBendingDofsOfNode(self, nodeNum: int) -> np.array:
        all_dofs = self.getDofsOfNode(nodeNum)
        return all_dofs[:3]
        
    def getMembraneDofsOfNode(self, nodeNum: int) -> np.array:
        all_dofs = self.getDofsOfNode(nodeNum)
        return all_dofs[3:]

    def getBendingDofsOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        bending_dofs = np.zeros((12,), dtype=int)
        for i, node in enumerate(nodesOfElement):
            bending_dofs[3*i : 3*(i+1)] = self.getBendingDofsOfNode(node)
        return bending_dofs

    def getMembraneDofsOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        membrane_dofs = np.zeros((8,), dtype=int)
        for i, node in enumerate(nodesOfElement):
            membrane_dofs[2*i : 2*(i+1)] = self.getMembraneDofsOfNode(node)
        return membrane_dofs
    
    def getNodesOfElement(self, elementNum: int) -> np.array:
        assert elementNum <= self.getNumElements() - 1, f"Element number {elementNum} exceeds the maximum number of elements, which is {self.getNumElements()-1}"
        numNodesFromRowsBefore = (elementNum // self.getNumElements_x())*(self.getNumElements_x() + 1)
        numNodesBeforeThisElementInTheColumn = (elementNum % self.getNumElements_x())
        numNodesBeforeThisElement = numNodesFromRowsBefore + numNodesBeforeThisElementInTheColumn
        bottomLeftNodeOfThisElementInLocalOrder = numNodesBeforeThisElement
        topLeftNodeOfThisElementInLocalOrder = bottomLeftNodeOfThisElementInLocalOrder + 1
        topRightNodesOfThisElementInLocalOrder = topLeftNodeOfThisElementInLocalOrder + (self.getNumElements_x() + 1)
        bottomRightNodesOfThisElementInLocalOrder = topRightNodesOfThisElementInLocalOrder - 1
        return np.array([bottomLeftNodeOfThisElementInLocalOrder, topLeftNodeOfThisElementInLocalOrder, topRightNodesOfThisElementInLocalOrder, bottomRightNodesOfThisElementInLocalOrder])
    
    def getCoordinatesOfNodesOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        coordinatesOfNodesOfThisElementInLocalOrder = np.zeros((self.getNumNodesPerElement(), 2), dtype=float)
        for i, node in enumerate(nodesOfElement):
            coordinatesOfNodesOfThisElementInLocalOrder[i,:] = self.getCoordinatesOfNode(node)
        return coordinatesOfNodesOfThisElementInLocalOrder

    def getDofsOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        dofsOfThisElementInLocalOrder = np.zeros((self.getNumDofsPerElement(),), dtype=int)
        for i, node in enumerate(nodesOfElement):
            dofsOfThisElementInLocalOrder[self.num_dofs_per_node*i:self.num_dofs_per_node*(i+1)] = self.getDofsOfNode(node)
        return dofsOfThisElementInLocalOrder
        
    def getSizeOfElement(self, elementNum: int) -> np.array:
        coordinatesOfNodesOfThisElementInLocalOrder = self.getCoordinatesOfNodesOfElement(elementNum)
        a = coordinatesOfNodesOfThisElementInLocalOrder[1,0] - coordinatesOfNodesOfThisElementInLocalOrder[0,0]
        b = coordinatesOfNodesOfThisElementInLocalOrder[3,1] - coordinatesOfNodesOfThisElementInLocalOrder[0,1]

        return np.array([a,b])
    
    def getElementBounds(self, elementNum: int) -> np.array: #For load computations
        #Helper function for getRectangleBoundsInElement
        #Return bounds as [xmin, xmax, ymin, ymax]
        coordinatesOfNodesOfThisElementInLocalOrder = self.getCoordinatesOfNodesOfElement(elementNum)
        return np.array([coordinatesOfNodesOfThisElementInLocalOrder[0,0], coordinatesOfNodesOfThisElementInLocalOrder[1,0], coordinatesOfNodesOfThisElementInLocalOrder[0,1], coordinatesOfNodesOfThisElementInLocalOrder[3,1]])
    
    def partitionRectangleToElements(self, rectangleBounds: np.array) -> np.array: #For load computations
        #Return bounds as [xmin, xmax, ymin, ymax]
        x_lower_index = np.sum(self.xnodes <= rectangleBounds[0]) - 1
        x_upper_index = np.sum(self.xnodes < rectangleBounds[1])
        y_lower_index = np.sum(self.ynodes <= rectangleBounds[2]) - 1
        y_upper_index = np.sum(self.ynodes < rectangleBounds[3])

        intersecting_elements = (np.arange(x_lower_index, x_upper_index) + np.arange(y_lower_index, y_upper_index)[:, np.newaxis]*self.getNumElements_x()).flatten() #Thanks, Gemini!
        
        return intersecting_elements
    
    def intervalOverlap(self, x1: np.array, x2: np.array) -> np.array:
        #Helper function for getRectangleBoundsInElement
        overlap_min = max(x1[0], x2[0])
        overlap_max = min(x1[1], x2[1])

        if overlap_min <= overlap_max:
            return np.array([overlap_min, overlap_max])
        else:
            return np.array([])
    
    def getRectangleBoundsInElement(self, elementNum: int, rectangleBounds: np.array) -> np.array: #For load computations
        #Return overlap bounds as [xmin, xmax, ymin, ymax]
        elementBounds = self.getElementBounds(elementNum) #Bounds as [xmin, xmax, ymin, ymax]
        x_overlap = self.intervalOverlap(elementBounds[0:2], rectangleBounds[0:2])
        y_overlap = self.intervalOverlap(elementBounds[2:4], rectangleBounds[2:4])

        assert (x_overlap.size != 0) and (x_overlap.size != 0), f"Error in finding the bounds of the load within element number {elementNum}!"

        return np.concatenate((x_overlap, y_overlap))
    
    def getLocalRectangleBoundsInElement(self, elementNum: int, rectangleBounds: np.array) -> np.array: #For load computations
        #Returns overlap bounds as [xmin, xmax, ymin, ymax] but with respect to the local coordinates of the element i.e., bottom left is (0,0)
        elementBounds = self.getElementBounds(elementNum) #Bounds as [xmin, xmax, ymin, ymax]
        overlapGlobal = self.getRectangleBoundsInElement(elementNum, rectangleBounds)
        overlapLocal = overlapGlobal
        overlapLocal[0:2] = overlapLocal[0:2] - elementBounds[0]
        overlapLocal[2:4] = overlapLocal[2:4] - elementBounds[2]
        return overlapLocal

    def test(self):
        print('**** Testing the mesh class')
        for ielem in range(self.getNumElements()):
            print('Element DOFs: ', ielem, self.getDofsOfElement(ielem))
            print('Element size:', ielem, self.getSizeOfElement(ielem))
            print('Element node coords: ', ielem, self.getCoordinatesOfNodesOfElement(ielem))
            for node in self.getNodesOfElement(ielem):
                print("Node and coords: ", node, self.getCoordinatesOfNode(node))
                print("Node and DOFs: ", node, self.getDofsOfNode(node))
            print('****')

class RectangleLoad:
    def __init__(self,q,rectangleBounds):
        self.q = q
        self.rectangleBounds = rectangleBounds #In order of [xmin, xmax, ymin, ymax] of rectangle

        assert rectangleBounds[1]>rectangleBounds[0], f"Error in load definition: x2 = {self.rectangleBounds[1]} < x1 = {self.rectangleBounds[0]}. The load must always be defined as x2 > x1."
        assert rectangleBounds[3]>rectangleBounds[2], f"Error in load definition: y2 = {self.rectangleBounds[3]} < y1 = {self.rectangleBounds[2]}. The load must always be defined as y2 > y1."
        assert q>0.0, f"Error in load definition: pressure q = {self.q} is not positive."

    def getLoadArea(self) -> float:
        return (self.rectangleBounds[1] - self.rectangleBounds[0])*(self.rectangleBounds[3] - self.rectangleBounds[2])
    
    def getLoadMagnitude(self) -> float:
        return self.getLoadArea()*self.q
    
    def test_load(self):
        print(f"The load has a pressure of {self.q} in bounds of {self.rectangleBounds}. The area of this load if {self.getLoadArea()}, and so the total load is {self.getLoadMagnitude()}.")

class SlabProperties:
    def __init__(self, Emod_, nu_, t_):
        self.Emod = Emod_
        self.nu = nu_
        self.t = t_
    
    def test(self):
        print(f"Elastic modulus = {self.Emod}, Poisson ratio = {self.nu}, Thickness = {self.t}")

class SubgradeProperties:
    def __init__(self, Kx, Ky, Kz):
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz
    
    def test(self):
        print(f"Kx = {self.Kx}, Ky = {self.Ky}, Kz = {self.Kz}")

def read_mesh_inputs(filename: str, ndofs_per_node: int) -> mesh_plate_enumx:
    try:
        with open(filename, 'r') as jsonFS:
            jsonData = json.load(jsonFS)
            xnodes = np.array(jsonData["nodes"]["x"], dtype=float)
            ynodes = np.array(jsonData["nodes"]["y"], dtype=float)
            return mesh_plate_enumx(xnodes, ynodes, ndofs_per_node)
    except Exception as e:
        print(f"An unexpected error occurred: {e}"); exit()

def read_slab_properties(filename: str) -> SlabProperties:
    try:
        with open(filename, 'r') as jsonFS:
            jsonData = json.load(jsonFS)
            Emod = jsonData["slab"]["Emod"]
            nu = jsonData["slab"]["nu"]
            t = jsonData["slab"]["t"]
            return SlabProperties(Emod, nu, t)
    except Exception as e:
        print(f"An unexpected error occurred: {e}"); exit()

def read_subgrade_properties(filename: str) -> SubgradeProperties:
    try:
        with open(filename, 'r') as jsonFS:
            jsonData = json.load(jsonFS)
            Kx = jsonData["subgrade"]["Kx"]
            Ky = jsonData["subgrade"]["Ky"]
            Kz = jsonData["subgrade"]["Kz"]
            return SubgradeProperties(Kx,Ky,Kz)
    except Exception as e:
        print(f"An unexpected error occurred: {e}"); exit()

def read_loads(filename: str) -> list:
    try:
        with open(filename, 'r') as jsonFS:
            jsonData = json.load(jsonFS)
            loads = []
            for load in jsonData["loads"]:
                bounds = np.array([load["x1"], load["x2"], load["y1"], load["y2"]])
                loads.append(RectangleLoad(load["q"], bounds))

        return loads

    except Exception as e:
        print(f"Error reading load file: {e}"); exit()

def test_partition(rectangle_load):
    print(f'Rectangle is: {rectangle_load}')
    intersecting_elements = mesh.partitionRectangleToElements(rectangle_load)
    print(f'Intersecting elements: {intersecting_elements}')

    for element in intersecting_elements:
        rect_coords = mesh.getRectangleBoundsInElement(element, rectangle_load)
        print(f'In element {element}, the load is over {rect_coords}')
    
    print('***')

def post_processor(mesh: mesh_plate_enumx, slab_properties: SlabProperties, Ug: np.array):
    nodal_stresses_top = np.zeros((mesh.getNumNodes(),3)) #sxx, syy, sxy
    nodal_stresses_bottom = np.zeros((mesh.getNumNodes(),3)) #sxx, syy, sxy
    nodal_stresses_membrane = np.zeros((mesh.getNumNodes(),3)) #sxx, syy, sxy
    nodal_strains_top = np.zeros((mesh.getNumNodes(),3)) #exx, eyy, exy
    nodal_strains_bottom = np.zeros((mesh.getNumNodes(),3)) #exx, eyy, exy
    nodal_strains_membrane = np.zeros((mesh.getNumNodes(),3)) #exx, eyy, exy
    nodal_counts = np.zeros((mesh.getNumNodes(),))
    D_b = Dmatbending(slab_properties.Emod, slab_properties.nu)
    D_m = Dmatmembrane(slab_properties.Emod, slab_properties.nu)

    #Loop over each element
    for ielem in range(mesh.getNumElements()):
        #Get info about the element (needed for B matrices)
        dofsOfElement = mesh.getDofsOfElement(ielem)
        elementSize = mesh.getSizeOfElement(ielem)
        nodesOfElement = mesh.getNodesOfElement(ielem)
        coordsOfNodesOfElement = mesh.getCoordinatesOfNodesOfElement(ielem)
        Ue = Ug[dofsOfElement].reshape(mesh.getNumNodesPerElement(), mesh.num_dofs_per_node) #Reshape is required to extract bending and membrane DOFs, there may be better ways to do it in Eigen
        Ue_bending = Ue[:,0:3].flatten() #First 3 DOFs are for bending
        Ue_membrane = Ue[:,3:5].flatten() #Last 2 DOFs are for membrane

        #Calculate strains and stresses at top and bottom surfaces at all four nodes in this element
        #Also count the number of times each node is visited for averaging later
        for i_loc, I_node in enumerate(nodesOfElement):
            localCoordinatesOfNode = coordsOfNodesOfElement[i_loc,:] - coordsOfNodesOfElement[0,:] #Bottom left node is at (0,0) locally
            B_b = Blocalbending(localCoordinatesOfNode[0], localCoordinatesOfNode[1], elementSize[0], elementSize[1])
            curvatures = B_b @ Ue_bending
            strains_top = -(slab_properties.t / 2.0) * curvatures
            stresses_top = D_b @ strains_top
            
            nodal_strains_top[I_node] += strains_top
            nodal_strains_bottom[I_node] += -strains_top #Opposite sign as top
            nodal_stresses_top[I_node] += stresses_top
            nodal_stresses_bottom[I_node] += -stresses_top #Opposite sign as top

            B_m = Blocalmembrane(localCoordinatesOfNode[0], localCoordinatesOfNode[1], elementSize[0], elementSize[1])
            strains_membrane = B_m @ Ue_membrane
            stresses_membrane = D_m @ strains_membrane
            nodal_strains_membrane[I_node] += strains_membrane
            nodal_stresses_membrane[I_node] += stresses_membrane

            nodal_counts[I_node] += 1
    
    #Average the nodal stresses and strains 
    assert np.all(nodal_counts != 0), "It appears as though at least one node was never visited during post-processing. Something is wrong."
    nodal_strains_top /= nodal_counts[:, np.newaxis]
    nodal_strains_bottom /= nodal_counts[:, np.newaxis]
    nodal_strains_membrane /= nodal_counts[:, np.newaxis]
    nodal_stresses_top /= nodal_counts[:, np.newaxis]
    nodal_stresses_bottom /= nodal_counts[:, np.newaxis]
    nodal_stresses_membrane /= nodal_counts[:, np.newaxis]
    
    nodal_strains_top += nodal_strains_membrane #Add membrane strains to bending stresses to get total stresses
    nodal_strains_bottom += nodal_strains_membrane #Add membrane strains to bending stresses to get total stresses
    nodal_stresses_top += nodal_stresses_membrane #Add membrane stresses to bending stresses to get total stresses
    nodal_strains_bottom += nodal_stresses_membrane #Add membrane stresses to bending stresses to get total stresses

    return nodal_strains_top, nodal_strains_bottom, nodal_strains_membrane, nodal_stresses_top, nodal_stresses_bottom, nodal_stresses_membrane

def export_to_vtk(mesh: mesh_plate_enumx, slab_properties: SlabProperties, Ug: np.array, filename="output.vtk"):
    # We pass (X, Y, Z) - remember your y is horizontal (X-screen), x is vertical (Y-screen)
    grid = pv.RectilinearGrid(mesh.ynodes, mesh.xnodes, np.array([0.0]))
    nodal_strains_top, nodal_strains_bottom, nodal_strains_membrane, nodal_stresses_top, nodal_stresses_bottom, nodal_stresses_membrane = post_processor(mesh, slab_properties, Ug)
    
    grid.point_data["w"] = Ug[0::5]  
    grid.point_data["thetax"] = Ug[1::5]
    grid.point_data["thetay"] = Ug[2::5]
    grid.point_data["u"] = Ug[3::5]
    grid.point_data["v"] = Ug[4::5]
    grid.point_data["top_sxx"] = nodal_stresses_top[:,0]
    grid.point_data["top_syy"] = nodal_stresses_top[:,1]
    grid.point_data["top_sxy"] = nodal_stresses_top[:,2]
    grid.point_data["bottom_sxx"] = nodal_stresses_bottom[:,0]
    grid.point_data["bottom_syy"] = nodal_stresses_bottom[:,1]
    grid.point_data["bottom_sxy"] = nodal_stresses_bottom[:,2]
    grid.point_data["top_exx"] = nodal_strains_top[:,0]
    grid.point_data["top_eyy"] = nodal_strains_top[:,1]
    grid.point_data["top_exy"] = nodal_strains_top[:,2]
    grid.point_data["bottom_exx"] = nodal_strains_bottom[:,0]
    grid.point_data["bottom_eyy"] = nodal_strains_bottom[:,1]
    grid.point_data["bottom_exy"] = nodal_strains_bottom[:,2]
    
    grid.save(filename)
    print(f"File saved: {filename}")

if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python your_script_name.py <input_file.json>"); exit()
    filename = argv[1]
    dofs_per_node = 5

    print('**** Reading mesh info')
    mesh = read_mesh_inputs(filename, dofs_per_node)
    print('Success!')

    print('**** Reading slab properties')
    slab_properties = read_slab_properties(filename)
    print('Success!')

    print('**** Reading subgrade properties')
    subgrade_properties = read_subgrade_properties(filename)
    print('Success!')

    print('**** Reading loads')
    loads = read_loads(filename)
    print('Success!')

    print('**** Assembling Global Stiffness Matrix (Kg)')
    #Build as LIL then cast to CSR for solving, similar implemetation in Eigen
    Kg = lil_matrix((mesh.getNumDofs(), mesh.getNumDofs()), dtype=float)
    for ielem in range(mesh.getNumElements()):
        dofs = mesh.getDofsOfElement(ielem)
        elementSize = mesh.getSizeOfElement(ielem)
        Ke = Klocal(slab_properties.Emod, slab_properties.nu, elementSize[0], elementSize[1], slab_properties.t, subgrade_properties.Kx, subgrade_properties.Ky , subgrade_properties.Kz )
        for i_loc, I in enumerate(dofs):
            for j_loc, J in enumerate(dofs):
                if True: #I >= J: #Since Kg is symmetric, store only the lower triangle. Python has no good way to solve this, so I am storing the whole thing here. Can be done better in Eigen!
                    Kg[I, J] += Ke[i_loc, j_loc]
    print('Success!')
    #spy(Kg); show()

    print("**** Assembling Global Force Vector (Fg)")
    Fg = np.zeros((mesh.getNumDofs(),))
    #Loop over each load
    for load in loads:
        print(f'Found load: {load.getLoadMagnitude()/1000.0} kN')
        #For the load, figure out which elements it overlaps with
        intersecting_elements = mesh.partitionRectangleToElements(load.rectangleBounds)

        #For each of those elements, figure out the overlapping bounds
        for elem in intersecting_elements:
            elementSize = mesh.getSizeOfElement(elem)
            localBoundsInElement = mesh.getLocalRectangleBoundsInElement(elem, load.rectangleBounds)
            
            #Create Fe and assemble into Fg
            Fe = Flocal(elementSize[0], elementSize[1], localBoundsInElement[0], localBoundsInElement[1], localBoundsInElement[2], localBoundsInElement[3], load.q)
            dofs = mesh.getDofsOfElement(elem)
            for i_loc, I in enumerate(dofs):
                Fg[I] += Fe[i_loc]
    print('Success!')

    print("**** Solving the system")
    Ug = spsolve(Kg.tocsr(), Fg)
    #print(f"Ug = {Ug.reshape(mesh.getNumNodes(), mesh.num_dofs_per_node)}") - OK
    #print(f"Fg = {Fg.reshape(mesh.getNumNodes(), mesh.num_dofs_per_node)}") - OK
    print('Success!')

    print("**** Post-processing: Calculating Nodal Stresses and Strains")
    #nodal_strains_top, nodal_strains_bottom, nodal_strains_membrane, nodal_stresses_top, nodal_stresses_bottom, nodal_stresses_membrane = post_processor(mesh, slab_properties, Ug)      
    print('Success!')

    print("**** Writing to file")
    filename = "output.vtk"
    export_to_vtk(mesh, slab_properties, Ug, filename)
    print('Success!')
    
    