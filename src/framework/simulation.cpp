#include "simulation.h"

#include "wrapper.h"

#include "metric.h"
#include "utils.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace ntt {
  auto stringifySimulationEngine(const SimulationEngine& sim) -> std::string {
    switch (sim) {
    case PICEngine:
      return "PIC";
    case SimulationEngine::GRPIC:
      return "GRPIC";
    default:
      return "N/A";
    }
  }
  auto stringifyBoundaryCondition(const BoundaryCondition& bc) -> std::string {
    switch (bc) {
    case BoundaryCondition::PERIODIC:
      return "Periodic";
    case BoundaryCondition::ABSORB:
      return "Absorbing";
    case BoundaryCondition::OPEN:
      return "Open";
    case BoundaryCondition::USER:
      return "User";
    case BoundaryCondition::COMM:
      return "Communicate";
    default:
      return "N/A";
    }
  }
  auto stringifyParticlePusher(const ParticlePusher& pusher) -> std::string {
    switch (pusher) {
    case ParticlePusher::BORIS:
      return "Boris";
    case ParticlePusher::VAY:
      return "Vay";
    case ParticlePusher::PHOTON:
      return "Photon";
    default:
      return "N/A";
    }
  }

  template <Dimension D, SimulationEngine S>
  Simulation<D, S>::Simulation(const toml::value& inputdata)
    : m_params { inputdata, D },
      problem_generator { m_params },
      meshblock { m_params.resolution(),
                  m_params.extent(),
                  m_params.metricParameters(),
                  m_params.species() },
      writer { m_params, meshblock },
      random_pool { constant::RandomSeed } {
    meshblock.random_pool_ptr = &random_pool;
    meshblock.boundaries      = m_params.boundaries();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Initialize() {
    // find timestep and effective cell size
    meshblock.setMinCellSize(meshblock.metric.findSmallestCell());
    meshblock.setTimestep(m_params.cfl() * meshblock.minCellSize());

    WaitAndSynchronize();
    PLOGD << "Simulation initialized.";
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::InitializeSetup() {
    problem_generator.UserInitFields(m_params, meshblock);
    problem_generator.UserInitParticles(m_params, meshblock);

    WaitAndSynchronize();
    PLOGD << "Setup initialized.";
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Verify() {
    meshblock.Verify();
    WaitAndSynchronize();
    PLOGD << "Prerun check passed.";
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::PrintDetails() {
    // !TODO: make this prettier
    PLOGI << "[Simulation details]";
    PLOGI << "   title: " << m_params.title();
    PLOGI << "   engine: " << stringifySimulationEngine(S);
    PLOGI << "   total runtime: " << m_params.totalRuntime();
    PLOGI << "   dt: " << meshblock.timestep() << " ["
          << static_cast<int>(m_params.totalRuntime() / meshblock.timestep()) << " steps]";

    PLOGI << "[domain]";
    PLOGI << "   dimension: " << static_cast<short>(D) << "D";
    PLOGI << "   metric: " << (meshblock.metric.label);

    if constexpr (S == SimulationEngine::GRPIC) {
      PLOGI << "   Spin parameter: " << (m_params.metricParameters()[3]);
    }

    std::string bc { "   boundary conditions: { " };
    for (auto& b : m_params.boundaries()) {
      bc += stringifyBoundaryCondition(b) + " x ";
    }
    bc.erase(bc.size() - 3);
    bc += " }";
    PLOGI << bc;

    std::string res { "   resolution: { " };
    for (auto& r : m_params.resolution()) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";
    PLOGI << res;

    std::string ext { "   extent: " };
    for (auto i { 0 }; i < (int)(m_params.extent().size()); i += 2) {
      ext += "{" + std::to_string(m_params.extent()[i]) + ", "
             + std::to_string(m_params.extent()[i + 1]) + "} ";
    }
    PLOGI << ext;

    std::string cell { "   cell size: " };
    cell += std::to_string(meshblock.minCellSize());
    PLOGI << cell;

    PLOGI << "[fiducial parameters]";
    PLOGI << "   ppc0: " << m_params.ppc0();
    PLOGI << "   rho0: " << m_params.larmor0() << " ["
          << m_params.larmor0() / meshblock.minCellSize() << " cells]";
    PLOGI << "   c_omp0: " << m_params.skindepth0() << " ["
          << m_params.skindepth0() / meshblock.minCellSize() << " cells]";
    PLOGI << "   sigma0: " << m_params.sigma0();

    if (meshblock.particles.size() > 0) {
      PLOGI << "[particles]";
      int i { 0 };
      for (auto& prtls : meshblock.particles) {
        PLOGI << "   [species #" << i + 1 << "]";
        PLOGI << "      label: " << prtls.label();
        PLOGI << "      mass: " << prtls.mass();
        PLOGI << "      charge: " << prtls.charge();
        PLOGI << "      pusher: " << stringifyParticlePusher(prtls.pusher());
        PLOGI << "      maxnpart: " << prtls.maxnpart() << " (" << prtls.npart() << ")";
        ++i;
      }
    } else {
      PLOGI << "[no particles]";
    }
    WaitAndSynchronize();
    PLOGD << "Simulation details printed.";
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::rangeActiveCells() -> range_t<D> {
    return meshblock.rangeActiveCells();
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::rangeAllCells() -> range_t<D> {
    return meshblock.rangeAllCells();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Finalize() {
    WaitAndSynchronize();
    PLOGD << "Simulation finalized.";
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::SynchronizeHostDevice() {
    WaitAndSynchronize();
    meshblock.SynchronizeHostDevice();
    // for (auto& species : meshblock.particles) {
    //   species.SynchronizeHostDevice();
    // }
    PLOGD << "... host-device synchronized";
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::PrintDiagnostics(std::ostream& os) {
    for (std::size_t i { 0 }; i < meshblock.particles.size(); ++i) {
      auto& species { meshblock.particles[i] };
      os << "species #" << i << ": " << species.npart() << " ("
         << (double)(species.npart()) * 100 / (double)(species.maxnpart()) << "%)\n";
    }
  }

}    // namespace ntt

#ifdef PIC_ENGINE
template class ntt::Simulation<ntt::Dim1, ntt::PICEngine>;
template class ntt::Simulation<ntt::Dim2, ntt::PICEngine>;
template class ntt::Simulation<ntt::Dim3, ntt::PICEngine>;
#elif defined(GRPIC_ENGINE)
template class ntt::Simulation<ntt::Dim2, ntt::SimulationEngine::GRPIC>;
template class ntt::Simulation<ntt::Dim3, ntt::SimulationEngine::GRPIC>;
#endif
