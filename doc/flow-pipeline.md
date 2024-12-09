# Flow simulation pipline flow
  This document dicepts the pipline flow of flow simulation present in this library.
  ## Initialize

  ## Update
  ~~~mermaid
  graph TD
  emitterAllocCS -->
  advectSinglePassCS_densityNoEmit_SST -->
  advectSinglePassCS_velocityNoEmit_SST -->
  emitterVelocityCS#1 -->
  emitterVelocityCS#2 -->
  updateLinearCS -->
  emitterDensityCS#1 -->
  emitterDensityCS#2 -->
  vorticityConfinementCS_noDensity -->
  divergenceCS -->
  jacobiCS#1 -->
  restrictCS#1 --> etc#1[...] -->
  prolongCS -->
  jacobiCS#n --> etc#2[...] -->
  subtractCS -->
  updateLinearCS#1 -->
  updateLinearCS#2 -->
  sparseFadeCS
  ~~~

  ## Non Single Pass Advect Update
  ~~~mermaid
  graph TD
  emitterAllocCS -->
  advectCS_SST#1 -->
  macCormackCS_densityNoEmit_SST -->
  advectCS_SST#2 -->
  macCormackCS_velocityNoEmit_SST -->
  emitterVelocityCS#1 -->
  emitterVelocityCS#2 -->
  updateLinearCS -->
  emitterDensityCS#1 -->
  emitterDensityCS#2 -->
  vorticityConfinementCS_noDensity -->
  divergenceCS -->
  jacobiCS#1 -->
  restrictCS#1 --> etc#1[...] -->
  prolongCS -->
  jacobiCS#n --> etc#2[...] -->
  subtractCS -->
  updateLinearCS#1 -->
  updateLinearCS#2 -->
  velocitySummaryCS -->
  densitySummaryCoarseCS -->
  blockManager1CS -->
  blockManager2CS -->
  sparseClearCS -->
  sparseScaleCS -->
  sparseDeallocateCS -->
  sparseFreeListCS -->
  sparseAllocateCS -->
  sparseBlockListCS -->
  sparseScaleCS#2 -->
  sparseDeallocateCS#2 -->
  sparseFreeListCS#2 -->
  sparseAllocateCS#2 -->
  sparseBlockListCS#2
  ~~~

  ## Render

