# Slab Codes
These are various codes developed at the Built Environment Lab for the analysis of **rigid pavement slabs on subgrade**. These are not meant for simply supported structural slabs. The codes are primarily in Python and C++, with other tools like WxMaxima for support. 

A summary of the codes:
- **MCZ20v:** An FEM model with 20 DOF rectagular MCZ elements for slabs on subgrade based on the MCZ formulation for thin plates integrated with an uncoupled membrane model.  The "v" in the name indicates that it is configured only for vertical (transverse) loads. Only a single slab with any number of rectangular loads can be analyzed. Visualization using ParaView and associated codes. Developed under a PM-ECRG grant.
- **MCZ20:** Similar to MCZ20v but includes both vertical and horizontal loads. Developed under a PM-ECRG grant.