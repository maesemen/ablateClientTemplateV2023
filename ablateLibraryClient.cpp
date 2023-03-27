#include <petsc.h>
#include <environment/runEnvironment.hpp>
#include <mathFunctions/functionFactory.hpp>
#include <memory>
#include "builder.hpp"
#include "domain/boxMesh.hpp"
#include "domain/modifiers/distributeWithGhostCells.hpp"
#include "domain/modifiers/ghostBoundaryCells.hpp"
#include "eos/perfectGas.hpp"
#include "eos/twoPhase.hpp"
#include "finiteVolume/boundaryConditions/essentialGhost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"
#include "finiteVolume/processes/surfaceForce.hpp"
#include "finiteVolume/fluxCalculator/riemannStiff.hpp"
#include "io/interval/fixedInterval.hpp"
#include "io/interval/simulationTimeInterval.hpp"
#include "io/hdf5Serializer.hpp"
#include "monitors/timeStepMonitor.hpp"
#include "parameters/mapParameters.hpp"
#include "utilities/petscUtilities.hpp"

typedef struct {
    PetscReal gamma1;
    PetscReal Rgas1;
    PetscReal gamma2;
    PetscReal Rgas2;
    PetscReal Radius;
    PetscReal deltaR;
    PetscReal rho1;
    PetscReal e1;
    PetscReal rho2;
    PetscReal e2;
    PetscReal u;
} InitialConditions;

static PetscErrorCode SetInitialConditionEuler(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if ((PetscSqr(x[0])+PetscSqr(x[1])) < PetscSqr(initialConditions->Radius)) {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho2;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rho2 * initialConditions->u;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOV] = initialConditions->rho2 * initialConditions->u;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = initialConditions->rho2 * initialConditions->e2;

    } else {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho1;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rho1 * initialConditions->u;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOV] = initialConditions->rho1 * initialConditions->u;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = initialConditions->rho1 * initialConditions->e1;
    }

    if (((PetscSqr(x[0])+PetscSqr(x[1])) - PetscSqr(initialConditions->Radius)) < (initialConditions->deltaR * 10)) {
        PetscReal cellRho=0, cellRhoE=0, cellx[100], celly[100];
        cellx[0] = x[0] - initialConditions->deltaR/2 + initialConditions->deltaR/200;
        celly[0] = x[1] - initialConditions->deltaR/2 + initialConditions->deltaR/200;
        for (int n = 1; n<100; ++n) {
            cellx[n] = cellx[n-1] + initialConditions->deltaR/100;
            celly[n] = celly[n-1] + initialConditions->deltaR/100;
        }
        for (int k = 0; k<100; ++k) {
            for (int l = 0; l<100; ++l) {
                if ((PetscSqr(cellx[k])+PetscSqr(celly[l])) <= PetscSqr(initialConditions->Radius) ) {
                    cellRho+=initialConditions->rho2;
                    cellRhoE+=initialConditions->rho2 * initialConditions->e2;
                } else {
                    cellRho+=initialConditions->rho1;
                    cellRhoE+=initialConditions->rho1 * initialConditions->e1;
                }
            }
        }
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = cellRho/100/100;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = cellRhoE/100/100;
    }

    return 0;
}

static PetscErrorCode SetInitialConditionDensityVF(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if ((PetscSqr(x[0])+PetscSqr(x[1])) < PetscSqr(initialConditions->Radius)) {
        u[0] = 0.0;

    } else {
        u[0] = initialConditions->rho1;
    }

    if (((PetscSqr(x[0])+PetscSqr(x[1])) - PetscSqr(initialConditions->Radius)) < initialConditions->deltaR * 10) {
        PetscReal cell=0, cellx[100], celly[100];
        cellx[0] = x[0] - initialConditions->deltaR/2 + initialConditions->deltaR/200;
        celly[0] = x[1] - initialConditions->deltaR/2 + initialConditions->deltaR/200;
        for (int n = 1; n<100; ++n) {
            cellx[n] = cellx[n-1] + initialConditions->deltaR/100;
            celly[n] = celly[n-1] + initialConditions->deltaR/100;
        }
        for (int k = 0; k<100; ++k) {
            for (int l = 0; l<100; ++l) {
                if ((PetscSqr(cellx[k])+PetscSqr(celly[l])) <= PetscSqr(initialConditions->Radius) ) {
                    cell+=0.0;
                } else {
                    cell+=initialConditions->rho1;
                }
            }
        }
        u[0] = cell/100/100;
    }

    return 0;
}

static PetscErrorCode SetInitialConditionVF(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if ((PetscSqr(x[0])+PetscSqr(x[1])) < PetscSqr(initialConditions->Radius)) {
        u[0] = 0.0;

    } else {
        u[0] = 1.0;
    }

    if (((PetscSqr(x[0])+PetscSqr(x[1])) - PetscSqr(initialConditions->Radius)) < initialConditions->deltaR * 10) {
        PetscReal cell=0, cellx[100], celly[100];
        cellx[0] = x[0] - initialConditions->deltaR/2 + initialConditions->deltaR/200;
        celly[0] = x[1] - initialConditions->deltaR/2 + initialConditions->deltaR/200;
        for (int n = 1; n<100; ++n) {
            cellx[n] = cellx[n-1] + initialConditions->deltaR/100;
            celly[n] = celly[n-1] + initialConditions->deltaR/100;
        }
        for (int k = 0; k<100; ++k) {
            for (int l = 0; l<100; ++l) {
                if ((PetscSqr(cellx[k])+PetscSqr(celly[l])) <= PetscSqr(initialConditions->Radius) ) {
                    cell+=0.0;
                } else {
                    cell+=1.0;
                }
            }
        }
        u[0] = cell/100/100;
        if ( x[0] < -1.8 && x[0] > -2.0 && x[1] < 0 && x[1] > -0.15){
            u[0] = cell/100/100;
        }
    }

    return 0;
}

static PetscErrorCode PhysicsBoundary_Euler(PetscReal time, const PetscReal *c, const PetscReal *n, const PetscScalar *a_xI, PetscScalar *a_xG, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho1;
    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOU] = initialConditions->u;
    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOV] = initialConditions->u;
    a_xG[ablate::finiteVolume::CompressibleFlowFields::RHOE] = initialConditions->rho1 * initialConditions->e1;
    return 0;
    PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
    // initialize petsc and mpi
    ablate::environment::RunEnvironment::Initialize(&argc, &argv);
    ablate::utilities::PetscUtilities::Initialize();

    {
        // define some initial conditions
//        InitialConditions initialConditions{.gamma1 = 1.395, .Rgas1 = 259.84, .gamma2 = 1.43, .Rgas2 = 106.4, .Radius = 2.0, .deltaR = 0.25, .rho1 = 1.0995777621393386, .e1 = 230237.97468354428, .rho2 = 2.685284640171858, .e2 = 86604.65116279072 , .p = 100000.0, .T = 350.0, .u = 0.0};
        InitialConditions initialConditions{.gamma1 = 1.4, .Rgas1 = 287.0, .gamma2 = 1.4, .Rgas2 = 287.0, .Radius = 0.02, .deltaR = 0.001, .rho1 = 500.0, .e1 = 500.0, .rho2 = 1000.0, .e2 = 250.00295125, .u = 0.0};
        // setup the run environment deltaR 30 = 0.0020689655172413794
        ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"title", "bubbleGasTest"}});
        ablate::environment::RunEnvironment::Setup(runEnvironmentParameters);

        auto eosOx = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma","1.4"},{"Rgas","287.0"}}));
        auto eosBz = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma","1.4"},{"Rgas","287.0"}}));
        auto eosTwoPhase = std::make_shared<ablate::eos::TwoPhase>(eosOx,eosBz);

        // determine required fields for finite volume compressible flow
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eosOx),
            std::make_shared<ablate::domain::FieldDescription>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
                std::make_shared<ablate::domain::FieldDescription>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
                    std::make_shared<ablate::domain::FieldDescription>("pressure", "",ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

        auto interval = std::make_shared<ablate::io::interval::FixedInterval>(0);
        auto intervalIO = std::make_shared<ablate::io::interval::SimulationTimeInterval>(0.01);

        auto domain =
            std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                      fieldDescriptors,
                                                      std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                        std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},
                                                      std::vector<int>{60, 60},
                                                      std::vector<double>{-0.03, -0.03}, // -0.0310344827586206897
                                                      std::vector<double>{0.03, 0.03},
                                                      std::vector<std::string>{"NONE","NONE"} /*boundary*/,
                                                      false /*simplex*/,
                                                      ablate::parameters::MapParameters::Create({{"dm_refine", "0"}, {"dm_distribute", ""}}));

        // Set up the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".5"}});

        // Set the initial conditions for euler
        auto initialConditionAll = {std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, ablate::mathFunctions::Create(SetInitialConditionEuler, (void *)&initialConditions)),
            std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, ablate::mathFunctions::Create(SetInitialConditionDensityVF, (void *)&initialConditions)),
                std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, ablate::mathFunctions::Create(SetInitialConditionVF, (void *)&initialConditions))};

        // create a time stepper
        auto serializer = std::make_shared<ablate::io::Hdf5Serializer>(intervalIO);
        auto timeStepper = ablate::solver::TimeStepper(
            domain, ablate::parameters::MapParameters::Create({{"ts_type","rk"},{"ts_dt", "0.00001"}, {"ts_adapt_type", "physicsConstrained"}, {"ts_max_time","0.5001"}}), {serializer}, {initialConditionAll});

        auto labelIds = std::vector<int>{1,2,3,4};
        auto boundaryConditions = std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{
            std::make_shared<ablate::finiteVolume::boundaryConditions::Ghost>("euler","wall", labelIds, PhysicsBoundary_Euler, (void *)&initialConditions),
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("rhoVF wall", labelIds, std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, ablate::mathFunctions::Create(initialConditions.rho1))),
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("vf wall", labelIds, std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, ablate::mathFunctions::Create("1.0")))};

        auto riemannFluxGG = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosOx,eosOx);
        auto riemannFluxGL = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosOx,eosBz);
        auto riemannFluxLG = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosBz,eosOx);
        auto riemannFluxLL = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosBz,eosBz);
        auto processes = std::vector<std::shared_ptr<ablate::finiteVolume::processes::Process>>{
            std::make_shared<ablate::finiteVolume::processes::SurfaceForce>(0.02361),
            std::make_shared<ablate::finiteVolume::processes::TwoPhaseEulerAdvection>(eosTwoPhase, parameters, riemannFluxGG, riemannFluxGL, riemannFluxLG, riemannFluxLL)};


        // Create a shockTube solver
        auto finiteVolumeSolver = std::make_shared<ablate::finiteVolume::FiniteVolumeSolver>("multiphaseSolve",
                                                                                              ablate::domain::Region::ENTIREDOMAIN,
                                                                                              nullptr, /*parameters*/
                                                                                              processes,
                                                                                              boundaryConditions /*boundary conditions*/);

        // register the flowSolver with the timeStepper
        timeStepper.Register(
            finiteVolumeSolver,
            {std::make_shared<ablate::monitors::TimeStepMonitor>(nullptr, interval)});

        // Solve the time stepper
        timeStepper.Solve();
    }

    ablate::environment::RunEnvironment::Finalize();
    return 0;
}