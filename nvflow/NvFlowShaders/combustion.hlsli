/*
 * Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef COMBUSTION_H
#define COMBUSTION_H

//! This is the include for before the constant buffer

struct CombustionParams {
  NvFlowFloat3 gravity;
  float burnPerFuel;

  float ignitionTemp;
  float burnPerTemp;
  float fuelPerBurn;
  float tempPerBurn;

  float smokePerBurn;
  float divergencePerBurn;
  float buoyancyPerTemp;
  float coolingRate;

  NvFlowFloat4 deltaTime;
};

#else

//! This is the include for after the constant buffer

float4 combustSimulate(float4 density4, float4 coarseDensity4) {
  float temp = density4.x;
  float fuel = density4.y;
  float burn = 0.f;
  float smoke = density4.w;

  float coarseTemp = coarseDensity4.x;
  float burnTemp = max(coarseTemp, temp);

  // this is using normalized temperature
  //  A temperature of 0.f means neutral buoyancy
  //  A temperature of 1.f has the maximum burn rate
  if (burnTemp >= combust.ignitionTemp && fuel > 0.f) {
    // burn rate is proportional to temperature beyond ignition temperature
    burn = combust.burnPerTemp * (saturate(burnTemp) - combust.ignitionTemp) *
           combust.deltaTime.z;

    // limit burn based on available fuel
    burn = min(burn, combust.burnPerFuel * fuel);

    // update fuel consumption based on burn
    fuel -= combust.fuelPerBurn * burn;

    // update temperature based on burn
    temp += combust.tempPerBurn * burn;

    // update smoke based on burn
    smoke += combust.smokePerBurn * burn;
  }

  // approximate cooling with damping (instead of solving the heat equation)
  { temp -= combust.coolingRate * combust.deltaTime.x * temp; }

  // limit maximum temperature for stability
  { temp = clamp(temp, -1.f, 1.f); }

  return float4(temp, fuel, burn, smoke);
}

float4 combustVelocity(float4 velocity4, float4 density4) {
  float temp = density4.x;
  float burn = density4.z;

  // apply buoyancy
  velocity4.xyz -= combust.deltaTime.x * combust.buoyancyPerTemp * temp *
                   combust.gravity.xyz;

  // generate a divergence offset for the pressure solver, place on w velocity
  // channel
  velocity4.w = combust.divergencePerBurn * burn;

  return velocity4;
}

#endif
