import numpy as np
from math import floor
from math import ceil

# Simulation Box Settings
Nx = 8
Ny = 8
Nz = 8
regSize = 5
Rx = ceil( Nx / regSize )
Ry = ceil( Ny / regSize )
Rz = ceil( Nz / regSize )

def findCellFromPos ( pos : int ) :
	'''
	pos : global position index encoding type and unit cell
	----------
	returns unit cell index
    '''
	return ((floor(pos/12)) % (Nx * Ny * Nz))

def findCoordsFromPos ( pos : int ) :
	'''
	pos : global position index encoding type and unit cell
	----------
	returns x, y, z coordinates of unit cell
	'''
	cell = findCellFromPos(pos)
	nz = floor(cell/(Nx * Ny))
	plane = cell % (Nx * Ny)
	ny = floor(plane/Nx)
	nx = plane % Nx
	return nx, ny, nz

def findCellFromCoords ( nx : int, ny : int, nz : int ) :
	'''
	nx, ny, nz : x, y, z coordinates of unit cell
	----------
	returns the unit cell index
	'''
	nx = nx % Nx
	ny = ny % Ny
	nz = nz % Nz
	return (nz * Nx * Ny) + (ny * Nx) + nx

def findPosFromType ( cell : int, type: int ) :
	'''
	cell : unit cell index 
	type : integer indexing the type of interstitial (see pdf document)
	----------
	returns the global position index of the interstitial site
	'''
	type = type % 12
	return (cell * 12 + type)

def isOctahedral ( pos : int ) :
	'''
	pos : global position index encoding type and unit cell
	----------
	returns true iff the site is an octahedral
	'''
	return ((pos % 12) <= 3)

def findRegionFromPos ( pos : int ) :
	'''
	pos : global position index encoding type and unit cell
	----------
	returns the region coordinates
	'''
	nx, ny, nz = findCoordsFromPos(pos)
	rx = floor(nx/regSize)
	ry = floor(ny/regSize)
	rz = floor(nz/regSize)
	return rx, ry, rz

def OctNeighbors ( pos : int ) :
	'''
	pos : global position index encoding type and unit cell
	----------
	returns a numpy array of global indices of nearest neighbors
	'''
	type = pos % 12
	nx, ny, nz = findCoordsFromPos(pos)
	if (type == 0):
		mx = (nx - 1) % Nx
		my = (ny - 1) % Ny
		mz = (nz - 1) % Nz
		neighbors = np.array(
			[
				# OCTAHEDRAL NEIGHBORS
				pos + 1,
				pos + 2,
				pos + 3, 
				findPosFromType(findCellFromCoords(mx, ny, nz),1),
				findPosFromType(findCellFromCoords(mx, ny, nz),2),
				findPosFromType(findCellFromCoords(nx, my, nz),1),
				findPosFromType(findCellFromCoords(nx, my, nz),3),
				findPosFromType(findCellFromCoords(nx, ny, mz),2),
				findPosFromType(findCellFromCoords(nx, ny, mz),3),
				findPosFromType(findCellFromCoords(nx, my, mz),3),
				findPosFromType(findCellFromCoords(mx, ny, mz),2),
				findPosFromType(findCellFromCoords(mx, my, nz),1),
				# TETRAHEDRAL NEIGHBORS
				pos + 4, 
				pos + 5, 
				pos + 6, 
				pos + 7, 
				pos + 8, 
				pos + 9, 
				pos + 10, 
				pos + 11
			]
		)
	if (type == 1):
		mx = (nx + 1) % Nx
		my = (ny + 1) % Ny
		mz = (nz - 1) % Nz
		neighbors = np.array(
			[
				# OCTAHEDRAL NEIGHBORS
				pos - 1, 
				pos + 1, 
				pos + 2,
				findPosFromType(findCellFromCoords(mx, ny, nz),0),
				findPosFromType(findCellFromCoords(mx, ny, nz),3),
				findPosFromType(findCellFromCoords(nx, my, nz),0),
				findPosFromType(findCellFromCoords(nx, my, nz),2),
				findPosFromType(findCellFromCoords(nx, ny, mz),2),
				findPosFromType(findCellFromCoords(nx, ny, mz),3),
				findPosFromType(findCellFromCoords(nx, my, mz),2),
				findPosFromType(findCellFromCoords(mx, ny, mz),3),
				findPosFromType(findCellFromCoords(mx, my, nz),0),
				# TETRAHEDRAL NEIGHBORS
				pos - type + 7, 
				pos - type + 11,
				findPosFromType(findCellFromCoords(mx, ny, nz),5),
				findPosFromType(findCellFromCoords(mx, ny, nz),9),
				findPosFromType(findCellFromCoords(nx, my, nz),6),
				findPosFromType(findCellFromCoords(nx, my, nz),10),
				findPosFromType(findCellFromCoords(mx, my, nz),4),
				findPosFromType(findCellFromCoords(mx, my, nz),8)
			]
		)
	if (type == 2):
		mx = (nx + 1) % Nx
		my = (ny - 1) % Ny
		mz = (nz + 1) % Nz
		neighbors = np.array(
			[
				# OCTAHEDRAL NEIGHBORS
				pos - 2, 
				pos - 1, 
				pos + 1,
				findPosFromType(findCellFromCoords(mx, ny, nz),0),
				findPosFromType(findCellFromCoords(mx, ny, nz),3),
				findPosFromType(findCellFromCoords(nx, my, nz),1),
				findPosFromType(findCellFromCoords(nx, my, nz),3),
				findPosFromType(findCellFromCoords(nx, ny, mz),0),
				findPosFromType(findCellFromCoords(nx, ny, mz),1),
				findPosFromType(findCellFromCoords(nx, my, mz),1),
				findPosFromType(findCellFromCoords(mx, ny, mz),0),
				findPosFromType(findCellFromCoords(mx, my, nz),3),
				# TETRAHEDRAL NEIGHBORS
				pos - type + 10, 
				pos - type + 11,
				findPosFromType(findCellFromCoords(mx, ny, nz),8),
				findPosFromType(findCellFromCoords(mx, ny, nz),9),
				findPosFromType(findCellFromCoords(nx, ny, mz),6),
				findPosFromType(findCellFromCoords(nx, ny, mz),7),
				findPosFromType(findCellFromCoords(mx, ny, mz),4),
				findPosFromType(findCellFromCoords(mx, ny, mz),5)
			]
		)
	if (type == 3):
		mx = (nx - 1) % Nx
		my = (ny + 1) % Ny
		mz = (nz + 1) % Nz
		neighbors = np.array (
			[
				# OCTAHEDRAL NEIGHBORS
				pos - 3, 
				pos - 2, 
				pos - 1,
				findPosFromType(findCellFromCoords(mx, ny, nz),1),
				findPosFromType(findCellFromCoords(mx, ny, nz),2),
				findPosFromType(findCellFromCoords(nx, my, nz),0),
				findPosFromType(findCellFromCoords(nx, my, nz),2),
				findPosFromType(findCellFromCoords(nx, ny, mz),0),
				findPosFromType(findCellFromCoords(nx, ny, mz),1),
				findPosFromType(findCellFromCoords(nx, my, mz),0),
				findPosFromType(findCellFromCoords(mx, ny, mz),1),
				findPosFromType(findCellFromCoords(mx, my, nz),2),
				# TETRAHEDRAL NEIGHBORS
				pos - type + 9, 
				pos - type + 11,
				findPosFromType(findCellFromCoords(nx, my, nz),8),
				findPosFromType(findCellFromCoords(nx, my, nz),10),
				findPosFromType(findCellFromCoords(nx, ny, mz),5),
				findPosFromType(findCellFromCoords(nx, ny, mz),7),
				findPosFromType(findCellFromCoords(nx, my, mz),4),
				findPosFromType(findCellFromCoords(nx, my, mz),6)
			]
		)
	return neighbors

def TetNeighbors ( pos : int ) :
	type = pos % 12
	nx, ny, nz = findCoordsFromPos(pos)
	neighbors = np.array([pos - type])
	if (type == 4):
		mx = (nx - 1) % Nx
		my = (ny - 1) % Ny
		mz = (nz - 1) % Nz
	if (type == 5):
		mx = (nx - 1) % Nx
		my = ny
		mz = (nz - 1) % Nz
	if (type == 6):
		mx = nx
		my = (ny - 1) % Ny
		mz = (nz - 1) % Nz
	if (type == 7):
		mx = nx
		my = ny
		mz = (nz - 1) % Nz
	if (type == 8):
		mx = (nx - 1) % Nx
		my = (ny - 1) % Ny
		mz = nz
	if (type == 9):
		mx = (nx - 1) % Nx
		my = ny
		mz = nz
	if (type == 10):
		mx = nx
		my = (ny - 1) % Ny
		mz = nz
	if (type == 11):
		mx = nx
		my = ny
		mz = nz
	neighbors = np.append(
		neighbors, 
		[
			findPosFromType(findCellFromCoords(mx, my, nz),1), 
			findPosFromType(findCellFromCoords(mx, ny, mz),2), 
			findPosFromType(findCellFromCoords(nx, my, mz),3)
		]
	)
	return neighbors
	
def findPosesFromCell ( cell : int ) :
	"""
 	eats cell index and spits out all 12 sites contained in that cell
 	"""
	return [
			cell * 12, 
			cell * 12 + 1, 
			cell * 12 + 2, 
			cell * 12 + 3, 
			cell * 12 + 4, 
			cell * 12 + 5, 
			cell * 12 + 6, 
			cell * 12 + 7, 
			cell * 12 + 8, 
			cell * 12 + 9, 
			cell * 12 + 10, 
			cell * 12 + 11, 
			]

def findPosesFromRegion ( rx : int, ry : int, rz : int ) :
	"""
 	eats region coordinates and spits out a list of all pos indices corresponding to the region
 	"""
	remx = Nx % regSize
	remy = Ny % regSize
	remz = Nz % regSize

	startx = regSize * rx 
	endx = regSize * (rx + 1) 
	starty = regSize * ry 
	endy = regSize * (ry + 1)
	startz = regSize * rz
	endz = regSize * (rz + 1)
	if ((remx != 0) and (rx == Rx - 1)): 
		endx = startx + remx
	if ((remy != 0) and (ry == Ry - 1)): 
		endy = starty + remy
	if ((remz != 0) and (rz == Rz - 1)): 
		endz = startz + remz
	print('x range:', startx, endx)
	print('y range:', starty, endy)
	print('z range:', startz, endz)
	poses = []
	for nz in range(startz, endz):
		for ny in range(starty, endy):
			for nx in range(startx, endx):
				poses = poses + findPosesFromCell(findCellFromCoords(nx,ny,nz))
	
	return poses
