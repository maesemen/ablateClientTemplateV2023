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
    PetscReal rho1;
    PetscReal e1;
    PetscReal rho2;
    PetscReal p;
    PetscReal T;
    PetscReal u;
} InitialConditions;

static PetscErrorCode SetInitialCondition(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx) {
    InitialConditions *initialConditions = (InitialConditions *)ctx;

    if (x[0] < 0.0) {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho1;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rho1 * initialConditions->u;

        PetscReal e = initialConditions->p / ((initialConditions->gamma1 - 1.0) * initialConditions->rho1);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->u);
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rho1;

    } else {
        u[ablate::finiteVolume::CompressibleFlowFields::RHO] = initialConditions->rho2;
        u[ablate::finiteVolume::CompressibleFlowFields::RHOU + 0] = initialConditions->rho2 * initialConditions->u;

        PetscReal e = initialConditions->p / ((initialConditions->gamma2 - 1.0) * initialConditions->rho2);
        PetscReal et = e + 0.5 * PetscSqr(initialConditions->u);
        u[ablate::finiteVolume::CompressibleFlowFields::RHOE] = et * initialConditions->rho2;
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
        InitialConditions initialConditions{.gamma1 = 1.395, .Rgas1 = 259.84, .gamma2 = 1.43, .Rgas2 = 106.4, .Radius = 2.0, .rho1 = 1.0995777621393386, .e1 = 230237.97468354428, .rho2 = 2.685284640171858, .p = 100000.0, .T = 350.0, .u = 0.0};

        // setup the run environment
        ablate::parameters::MapParameters runEnvironmentParameters(std::map<std::string, std::string>{{"title", "bubbleGasTest"}});
        ablate::environment::RunEnvironment::Setup(runEnvironmentParameters);

        auto eosOx = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma","1.395"},{"Rgas","259.84"}}));
        auto eosBz = std::make_shared<ablate::eos::PerfectGas>(std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"gamma","1.43"},{"Rgas","106.4"}}));
        auto eosTwoPhase = std::make_shared<ablate::eos::TwoPhase>(eosOx,eosBz);

        // determine required fields for finite volume compressible flow
        std::vector<std::shared_ptr<ablate::domain::FieldDescriptor>> fieldDescriptors = {std::make_shared<ablate::finiteVolume::CompressibleFlowFields>(eosOx),
            std::make_shared<ablate::domain::FieldDescription>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
                std::make_shared<ablate::domain::FieldDescription>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, "", ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::SOL, ablate::domain::FieldType::FVM),
                    std::make_shared<ablate::domain::FieldDescription>("pressure", "",ablate::domain::FieldDescription::ONECOMPONENT, ablate::domain::FieldLocation::AUX, ablate::domain::FieldType::FVM)};

        auto interval = std::make_shared<ablate::io::interval::FixedInterval>(0);

        auto domain =
            std::make_shared<ablate::domain::BoxMesh>("simpleMesh",
                                                      fieldDescriptors,
                                                      std::vector<std::shared_ptr<ablate::domain::modifiers::Modifier>>{std::make_shared<ablate::domain::modifiers::DistributeWithGhostCells>(),
                                                                                                                        std::make_shared<ablate::domain::modifiers::GhostBoundaryCells>()},
                                                      std::vector<int>{30, 30},
                                                      std::vector<double>{-3, -3},
                                                      std::vector<double>{3, 3},
                                                      std::vector<std::string>{"NONE","NONE"} /*boundary*/,
                                                      false /*simplex*/,
                                                      ablate::parameters::MapParameters::Create({{"dm_refine", "0"}, {"dm_distribute", ""}}));

        // Set up the flow data
        auto parameters = std::make_shared<ablate::parameters::MapParameters>(std::map<std::string, std::string>{{"cfl", ".5"}});

        // Set the initial conditions for euler
        auto initialConditionAll = {std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD, ablate::mathFunctions::Create(SetInitialCondition, (void *)&initialConditions)),
            std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, ablate::mathFunctions::Create("x < 0 ? 1 : 0")),
                std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, ablate::mathFunctions::Create("x < 0 ? 1 : 0"))};

        // create a time stepper
        auto serializer = std::make_shared<ablate::io::Hdf5Serializer>(interval);
        auto timeStepper = ablate::solver::TimeStepper(
            domain, ablate::parameters::MapParameters::Create({{"ts_adapt_type", "physicsConstrained"}, {"ts_max_time","1.0"},{"ts_max_steps", "10"}, {"ts_dt", "0.00001"}}), {serializer}, {initialConditionAll});

        auto labelIds = std::vector<int>{1,2,3,4};
        auto boundaryConditions = std::vector<std::shared_ptr<ablate::finiteVolume::boundaryConditions::BoundaryCondition>>{
            std::make_shared<ablate::finiteVolume::boundaryConditions::Ghost>("euler","wall", labelIds, PhysicsBoundary_Euler, (void *)&initialConditions),
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("rhoVF wall", labelIds, std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, ablate::mathFunctions::Create(initialConditions.rho1))),
            std::make_shared<ablate::finiteVolume::boundaryConditions::EssentialGhost>("vf wall", labelIds, std::make_shared<ablate::mathFunctions::FieldFunction>(ablate::finiteVolume::processes::TwoPhaseEulerAdvection::VOLUME_FRACTION_FIELD, ablate::mathFunctions::Create("1.0")))};

        auto riemannFluxGG = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosOx,eosOx);
        auto riemannFluxGL = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosOx,eosBz);
        auto riemannFluxLG = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosBz,eosOx);
        auto riemannFluxLL = std::make_shared<ablate::finiteVolume::fluxCalculator::RiemannStiff>(eosBz,eosBz);
        auto processes = std::vector<std::shared_ptr<ablate::finiteVolume::processes::Process>>{std::make_shared<ablate::finiteVolume::processes::TwoPhaseEulerAdvection>(eosTwoPhase, parameters, riemannFluxGG,
            riemannFluxGL, riemannFluxLG, riemannFluxLL)};
//        std::make_shared<ablate::finiteVolume::processes::SurfaceForce>()>()};

        // Create a shockTube solver
        auto finiteVolumeSolver = std::make_shared<ablate::finiteVolume::FiniteVolumeSolver>("multiphaseSolve",
                                                                                              ablate::domain::Region::ENTIREDOMAIN,
                                                                                              nullptr, /*parameters*/
                                                                                              processes,
                                                                                              boundaryConditions /*boundary conditions*/);

        // register the flowSolver with the timeStepper
        timeStepper.Register(
            finiteVolumeSolver,
            {std::make_shared<ablate::monitors::TimeStepMonitor>()});

        // Solve the time stepper
        timeStepper.Solve();
    }

    ablate::environment::RunEnvironment::Finalize();
    return 0;
}