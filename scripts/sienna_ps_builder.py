"""
Python-Julia Interface for PowerSimulations.jl
Allows running PowerSimulations.jl directly from Python using PyJulia
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import json

logger = logging.getLogger(__name__)

class JuliaPowerSimulationsInterface:
    """
    Interface to run PowerSimulations.jl from Python using PyJulia.
    
    This class handles:
    1. Julia environment setup
    2. PowerSystems.jl data import
    3. PowerSimulations.jl dispatch execution
    4. Results retrieval back to Python
    """
    
    def __init__(self, julia_project_path: Optional[str] = None):
        """
        Initialize Julia interface.
        
        Parameters:
        -----------
        julia_project_path : str, optional
            Path to Julia project with PowerSystems.jl and PowerSimulations.jl
        """
        self.julia_project_path = julia_project_path
        self.julia = None
        self.powersystems = None
        self.powersimulations = None
        self.system = None
        
        self._setup_julia()
    
    def _setup_julia(self):
        """Setup Julia environment and import required packages."""
        try:
            # Import PyJulia
            import julia
            from julia import Julia
            
            # Initialize Julia
            if self.julia_project_path:
                # Use specific project environment
                jl = Julia(compiled_modules=False)
                jl.eval(f'using Pkg; Pkg.activate("{self.julia_project_path}")')
            else:
                # Use default environment
                jl = Julia(compiled_modules=False)
            
            self.julia = jl
            
            # Import required Julia packages
            logger.info("Importing Julia packages...")
            self.julia.eval("using PowerSystems")
            self.julia.eval("using PowerSimulations") 
            self.julia.eval("using HiGHS")  # Default solver
            self.julia.eval("using Dates")
            self.julia.eval("using TimeSeries")
            
            # Get package references
            self.powersystems = self.julia.PowerSystems
            self.powersimulations = self.julia.PowerSimulations
            
            logger.info("✓ Julia environment setup complete")
            
        except ImportError:
            logger.error("PyJulia not found. Install with: pip install julia")
            raise
        except Exception as e:
            logger.error(f"Failed to setup Julia environment: {e}")
            logger.info("Ensure Julia is installed and PyJulia is configured correctly")
            logger.info("Run: python -c 'import julia; julia.install()'")
            raise
    
    def load_pypsa_system(self, data_dir: str, base_power: float = 100.0) -> Any:
        """
        Load PyPSA-exported data into PowerSystems.jl System.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing PyPSA-exported CSV files
        base_power : float
            Base power in MVA
            
        Returns:
        --------
        Julia System object
        """
        data_path = Path(data_dir).absolute()
        
        logger.info(f"Loading PyPSA data from: {data_path}")
        
        # Check required files exist
        required_files = ['user_descriptors.yaml', 'bus.csv']
        for file in required_files:
            file_path = data_path / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        try:
            # Create PowerSystemTableData in Julia
            self.julia.eval(f'data_dir = "{data_path}"')
            self.julia.eval(f'base_power = {base_power}')
            self.julia.eval('user_descriptors = joinpath(data_dir, "user_descriptors.yaml")')
            
            # Check for optional files
            ts_metadata_file = data_path / "timeseries_metadata.json"
            gen_mapping_file = data_path / "generator_mapping.yaml"
            
            if ts_metadata_file.exists() and gen_mapping_file.exists():
                self.julia.eval('''
                    data = PowerSystemTableData(
                        data_dir,
                        base_power,
                        user_descriptors;
                        timeseries_metadata_file = joinpath(data_dir, "timeseries_metadata.json"),
                        generator_mapping_file = joinpath(data_dir, "generator_mapping.yaml")
                    )
                ''')
            elif ts_metadata_file.exists():
                self.julia.eval('''
                    data = PowerSystemTableData(
                        data_dir,
                        base_power, 
                        user_descriptors;
                        timeseries_metadata_file = joinpath(data_dir, "timeseries_metadata.json")
                    )
                ''')
            else:
                self.julia.eval('''
                    data = PowerSystemTableData(
                        data_dir,
                        base_power,
                        user_descriptors
                    )
                ''')
            
            # Create System
            self.julia.eval('sys = System(data, time_series_in_memory = true)')
            self.system = self.julia.eval('sys')
            
            # Get system summary
            bus_count = self.julia.eval('length(get_components(Bus, sys))')
            gen_count = self.julia.eval('length(get_components(Generator, sys))')
            load_count = self.julia.eval('length(get_components(ElectricLoad, sys))')
            
            logger.info(f"✓ PowerSystems.jl System created:")
            logger.info(f"  Buses: {bus_count}")
            logger.info(f"  Generators: {gen_count}")
            logger.info(f"  Loads: {load_count}")
            
            return self.system
            
        except Exception as e:
            logger.error(f"Failed to load PyPSA system: {e}")
            raise
    
    def run_economic_dispatch(self, 
                            start_time: str = "2030-01-01T00:00:00",
                            horizon_days: int = 7,
                            solver: str = "HiGHS") -> Dict[str, Any]:
        """
        Run Economic Dispatch simulation.
        
        Parameters:
        -----------
        start_time : str
            Start time for simulation (ISO format)
        horizon_days : int
            Simulation horizon in days
        solver : str
            Solver to use (HiGHS, Gurobi, CPLEX, etc.)
            
        Returns:
        --------
        Dict with simulation results
        """
        if self.system is None:
            raise ValueError("No system loaded. Call load_pypsa_system() first.")
        
        logger.info(f"Running Economic Dispatch simulation...")
        logger.info(f"  Start time: {start_time}")
        logger.info(f"  Horizon: {horizon_days} days")
        logger.info(f"  Solver: {solver}")
        
        try:
            # Set up simulation parameters
            self.julia.eval(f'start_time = DateTime("{start_time}")')
            self.julia.eval(f'horizon = Day({horizon_days})')
            self.julia.eval(f'end_time = start_time + horizon')
            
            # Create Economic Dispatch template
            self.julia.eval('template = EconomicDispatchTemplate()')
            
            # Set up solver
            if solver == "HiGHS":
                self.julia.eval('optimizer = HiGHS.Optimizer')
            elif solver == "Gurobi":
                self.julia.eval('using Gurobi; optimizer = Gurobi.Optimizer')
            elif solver == "CPLEX":
                self.julia.eval('using CPLEX; optimizer = CPLEX.Optimizer')
            else:
                logger.warning(f"Unknown solver {solver}, using HiGHS")
                self.julia.eval('optimizer = HiGHS.Optimizer')
            
            # Create Decision Model
            self.julia.eval('''
                decision_model = DecisionModel(
                    template,
                    sys;
                    name = "PyPSA_ED",
                    optimizer = optimizer,
                    optimizer_solve_log_print = false
                )
            ''')
            
            # Solve
            self.julia.eval('solve!(decision_model)')
            
            # Check solve status
            solve_status = self.julia.eval('get_termination_status(decision_model)')
            logger.info(f"Solve status: {solve_status}")
            
            if str(solve_status) != "OPTIMAL":
                logger.warning(f"Solver did not find optimal solution: {solve_status}")
            
            # Get results
            self.julia.eval('results = OptimizationProblemResults(decision_model)')
            
            # Extract key results
            results = self._extract_results()
            
            logger.info("✓ Economic Dispatch completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Economic Dispatch failed: {e}")
            raise
    
    def run_unit_commitment(self,
                          start_time: str = "2030-01-01T00:00:00", 
                          horizon_days: int = 7,
                          solver: str = "HiGHS") -> Dict[str, Any]:
        """
        Run Unit Commitment simulation.
        
        Parameters:
        -----------
        start_time : str
            Start time for simulation (ISO format)
        horizon_days : int
            Simulation horizon in days
        solver : str
            Solver to use
            
        Returns:
        --------
        Dict with simulation results
        """
        if self.system is None:
            raise ValueError("No system loaded. Call load_pypsa_system() first.")
        
        logger.info(f"Running Unit Commitment simulation...")
        
        try:
            # Set up simulation parameters
            self.julia.eval(f'start_time = DateTime("{start_time}")')
            self.julia.eval(f'horizon = Day({horizon_days})')
            
            # Create Unit Commitment template
            self.julia.eval('template = UnitCommitmentTemplate()')
            
            # Set up solver
            if solver == "HiGHS":
                self.julia.eval('optimizer = HiGHS.Optimizer')
            elif solver == "Gurobi":
                self.julia.eval('using Gurobi; optimizer = Gurobi.Optimizer')
            else:
                logger.warning(f"Unknown solver {solver}, using HiGHS")
                self.julia.eval('optimizer = HiGHS.Optimizer')
            
            # Create Decision Model
            self.julia.eval('''
                decision_model = DecisionModel(
                    template,
                    sys;
                    name = "PyPSA_UC",
                    optimizer = optimizer,
                    optimizer_solve_log_print = false
                )
            ''')
            
            # Solve
            self.julia.eval('solve!(decision_model)')
            
            # Get results
            self.julia.eval('results = OptimizationProblemResults(decision_model)')
            
            results = self._extract_results()
            
            logger.info("✓ Unit Commitment completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Unit Commitment failed: {e}")
            raise
    
    def _extract_results(self) -> Dict[str, Any]:
        """Extract results from Julia and convert to Python format."""
        
        results = {}
        
        try:
            # Get objective value
            results['objective_value'] = float(self.julia.eval('get_objective_value(results)'))
            
            # Get generator dispatch results
            gen_results = self.julia.eval('''
                read_realized_variables(results, names = [:ActivePowerVariable])
            ''')
            
            # Convert Julia DataFrames to Python (this is simplified)
            # In practice, you might want to use DataFrames.jl to CSV export
            
            # Get variable names
            variable_names = self.julia.eval('names(gen_results)')
            
            # For now, return summary statistics
            results['generator_dispatch'] = {
                'variables': [str(name) for name in variable_names],
                'summary': 'Generator dispatch data available in Julia results object'
            }
            
            # Get dual variables (prices) if available
            try:
                dual_results = self.julia.eval('''
                    read_realized_duals(results, names = [:CopperPlateBalanceConstraint])
                ''')
                results['electricity_prices'] = {
                    'summary': 'Electricity price data available in Julia results object'
                }
            except:
                results['electricity_prices'] = {'summary': 'No dual variables available'}
            
            # Get solve time
            try:
                solve_time = self.julia.eval('get_solve_time(results)')
                results['solve_time_seconds'] = float(solve_time)
            except:
                results['solve_time_seconds'] = None
            
            logger.info(f"Results extracted: Objective = {results['objective_value']}")
            
        except Exception as e:
            logger.warning(f"Failed to extract some results: {e}")
            results['extraction_error'] = str(e)
        
        return results
    
    def export_results_to_csv(self, output_dir: str) -> Dict[str, str]:
        """
        Export simulation results to CSV files.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save result CSV files
            
        Returns:
        --------
        Dict with paths to created CSV files
        """
        if not hasattr(self, 'results') or self.julia.eval('isdefined(Main, :results)') == False:
            raise ValueError("No results available. Run a simulation first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = {}
        
        try:
            # Export generator dispatch
            self.julia.eval(f'results_dir = "{output_path.absolute()}"')
            self.julia.eval('mkpath(results_dir)')
            
            # Export realized variables
            self.julia.eval('''
                gen_dispatch = read_realized_variables(results, names = [:ActivePowerVariable])
                CSV.write(joinpath(results_dir, "generator_dispatch.csv"), gen_dispatch)
            ''')
            files_created['generator_dispatch'] = str(output_path / "generator_dispatch.csv")
            
            # Export dual variables (prices)
            try:
                self.julia.eval('''
                    prices = read_realized_duals(results, names = [:CopperPlateBalanceConstraint])
                    CSV.write(joinpath(results_dir, "electricity_prices.csv"), prices)
                ''')
                files_created['electricity_prices'] = str(output_path / "electricity_prices.csv")
            except:
                logger.warning("Could not export electricity prices")
            
            # Export system status (for Unit Commitment)
            try:
                self.julia.eval('''
                    status = read_realized_variables(results, names = [:OnVariable])
                    CSV.write(joinpath(results_dir, "unit_status.csv"), status)
                ''')
                files_created['unit_status'] = str(output_path / "unit_status.csv")
            except:
                logger.info("No unit commitment status available")
            
            # Export summary statistics
            summary_data = {
                'objective_value': float(self.julia.eval('get_objective_value(results)')),
                'solve_time': float(self.julia.eval('get_solve_time(results)')),
                'termination_status': str(self.julia.eval('get_termination_status(get_decision_model(results))'))
            }
            
            summary_file = output_path / "simulation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            files_created['summary'] = str(summary_file)
            
            logger.info(f"Results exported to {len(files_created)} files in {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise
        
        return files_created
    
    def compare_with_pypsa_results(self, pypsa_results_file: str) -> Dict[str, Any]:
        """
        Compare PowerSimulations.jl results with original PyPSA results.
        
        Parameters:
        -----------
        pypsa_results_file : str
            Path to PyPSA results NetCDF file
            
        Returns:
        --------
        Dict with comparison metrics
        """
        logger.info("Comparing PowerSimulations.jl results with PyPSA...")
        
        try:
            import pypsa
            
            # Load PyPSA results
            pypsa_network = pypsa.Network(pypsa_results_file)
            
            # Get PyPSA generator dispatch
            pypsa_dispatch = pypsa_network.generators_t.p
            pypsa_total_generation = pypsa_dispatch.sum().sum()
            pypsa_objective = getattr(pypsa_network, 'objective', None)
            
            # Get PowerSimulations.jl results
            sienna_objective = float(self.julia.eval('get_objective_value(results)'))
            
            # Basic comparison
            comparison = {
                'pypsa_total_generation_mwh': float(pypsa_total_generation),
                'pypsa_objective': float(pypsa_objective) if pypsa_objective else None,
                'sienna_objective': sienna_objective,
                'objective_difference': None,
                'objective_relative_error': None
            }
            
            if pypsa_objective:
                comparison['objective_difference'] = sienna_objective - pypsa_objective
                comparison['objective_relative_error'] = abs(comparison['objective_difference']) / abs(pypsa_objective)
            
            logger.info(f"Comparison complete:")
            logger.info(f"  PyPSA objective: {comparison['pypsa_objective']}")
            logger.info(f"  Sienna objective: {comparison['sienna_objective']}")
            if comparison['objective_relative_error']:
                logger.info(f"  Relative error: {comparison['objective_relative_error']:.4%}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {'error': str(e)}


class PyPSAToSiennaWorkflow:
    """
    Complete workflow for PyPSA to Sienna conversion and simulation.
    
    This class orchestrates:
    1. PyPSA network export to CSV
    2. Sienna system creation
    3. PowerSimulations.jl execution
    4. Results comparison
    """
    
    def __init__(self, julia_project_path: Optional[str] = None):
        """Initialize workflow with optional Julia project path."""
        self.sienna_interface = JuliaPowerSimulationsInterface(julia_project_path)
        self.export_results = None
        self.simulation_results = None
    
    def run_complete_workflow(self,
                            pypsa_network,
                            scenario_setup: dict,
                            output_base_dir: str,
                            simulation_type: str = "economic_dispatch",
                            start_time: str = "2030-01-01T00:00:00",
                            horizon_days: int = 7,
                            solver: str = "HiGHS") -> Dict[str, Any]:
        """
        Run the complete PyPSA to Sienna workflow.
        
        Parameters:
        -----------
        pypsa_network : pypsa.Network
            Solved PyPSA network
        scenario_setup : dict
            Scenario configuration
        output_base_dir : str
            Base directory for all outputs
        simulation_type : str
            Type of simulation ("economic_dispatch" or "unit_commitment")
        start_time : str
            Simulation start time
        horizon_days : int
            Simulation horizon
        solver : str
            Solver to use
            
        Returns:
        --------
        Dict with complete workflow results
        """
        from scripts.export_to_sienna_old import export_pypsa_to_sienna
        
        output_path = Path(output_base_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        workflow_results = {}
        
        try:
            # Step 1: Export PyPSA to Sienna CSV format
            logger.info("=== Step 1: Exporting PyPSA to Sienna CSV ===")
            
            csv_export_dir = output_path / "sienna_csv_data"
            self.export_results = export_pypsa_to_sienna(
                network=pypsa_network,
                scenario_setup=scenario_setup,
                output_dir=str(csv_export_dir),
                include_time_series=True
            )
            workflow_results['export_results'] = self.export_results
            logger.info("✓ PyPSA export complete")
            
            # Step 2: Load data into PowerSystems.jl
            logger.info("=== Step 2: Loading data into PowerSystems.jl ===")
            
            system = self.sienna_interface.load_pypsa_system(str(csv_export_dir))
            workflow_results['system_loaded'] = True
            logger.info("✓ PowerSystems.jl system created")
            
            # Step 3: Run simulation
            logger.info(f"=== Step 3: Running {simulation_type} simulation ===")
            
            if simulation_type == "economic_dispatch":
                self.simulation_results = self.sienna_interface.run_economic_dispatch(
                    start_time=start_time,
                    horizon_days=horizon_days,
                    solver=solver
                )
            elif simulation_type == "unit_commitment":
                self.simulation_results = self.sienna_interface.run_unit_commitment(
                    start_time=start_time,
                    horizon_days=horizon_days,
                    solver=solver
                )
            else:
                raise ValueError(f"Unknown simulation type: {simulation_type}")
            
            workflow_results['simulation_results'] = self.simulation_results
            logger.info("✓ Simulation complete")
            
            # Step 4: Export results
            logger.info("=== Step 4: Exporting simulation results ===")
            
            results_dir = output_path / "simulation_results"
            result_files = self.sienna_interface.export_results_to_csv(str(results_dir))
            workflow_results['result_files'] = result_files
            logger.info("✓ Results exported")
            
            # Step 5: Performance summary
            workflow_results['performance_summary'] = {
                'objective_value': self.simulation_results.get('objective_value'),
                'solve_time_seconds': self.simulation_results.get('solve_time_seconds'),
                'simulation_type': simulation_type,
                'horizon_days': horizon_days,
                'solver': solver
            }
            
            logger.info("=== Workflow Complete ===")
            logger.info(f"Objective value: {workflow_results['performance_summary']['objective_value']}")
            logger.info(f"Solve time: {workflow_results['performance_summary']['solve_time_seconds']} seconds")
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            workflow_results['error'] = str(e)
            raise


# Integration functions for solve_network_dispatch.py
def run_sienna_dispatch_from_pypsa(pypsa_network,
                                  scenario_setup: dict,
                                  output_dir: str,
                                  simulation_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Integration function for solve_network_dispatch.py.
    
    Call this when export_to_Sienna=True to run the complete workflow.
    
    Parameters:
    -----------
    pypsa_network : pypsa.Network
        Solved PyPSA dispatch network
    scenario_setup : dict
        Scenario configuration
    output_dir : str
        Output directory for all Sienna files
    simulation_config : dict, optional
        Simulation configuration with keys:
        - simulation_type: "economic_dispatch" or "unit_commitment" 
        - start_time: ISO datetime string
        - horizon_days: int
        - solver: "HiGHS", "Gurobi", etc.
        
    Returns:
    --------
    Dict with complete workflow results
    """
    # Default simulation configuration
    default_config = {
        'simulation_type': 'economic_dispatch',
        'start_time': '2030-01-01T00:00:00',
        'horizon_days': 7,
        'solver': 'HiGHS'
    }
    
    if simulation_config:
        default_config.update(simulation_config)
    
    # Run complete workflow
    workflow = PyPSAToSiennaWorkflow()
    
    results = workflow.run_complete_workflow(
        pypsa_network=pypsa_network,
        scenario_setup=scenario_setup,
        output_base_dir=output_dir,
        **default_config
    )
    
    return results


def setup_julia_environment():
    """
    Setup Julia environment for PowerSystems.jl and PowerSimulations.jl.
    
    Run this once to install required Julia packages.
    """
    logger.info("Setting up Julia environment...")
    
    try:
        import julia
        from julia import Julia
        
        # Install Julia packages
        jl = Julia(compiled_modules=False)
        jl.eval('using Pkg')
        jl.eval('Pkg.add(["PowerSystems", "PowerSimulations", "HiGHS", "CSV", "DataFrames"])')
        
        logger.info("✓ Julia environment setup complete")
        
    except ImportError:
        logger.error("PyJulia not found. Install with: pip install julia")
        logger.info("Then run: python -c 'import julia; julia.install()'")
        raise


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("PyPSA-Sienna Integration Interface")
    print("==================================")
    print()
    print("This module provides integration between PyPSA and PowerSimulations.jl")
    print()
    print("Usage:")
    print("1. Setup Julia environment (run once):")
    print("   from python_julia_interface import setup_julia_environment")
    print("   setup_julia_environment()")
    print()
    print("2. Run complete workflow:")
    print("   from python_julia_interface import run_sienna_dispatch_from_pypsa")
    print("   results = run_sienna_dispatch_from_pypsa(network, scenario_setup, './output')")
    print()
    print("3. Or use step-by-step interface:")
    print("   interface = JuliaPowerSimulationsInterface()")
    print("   system = interface.load_pypsa_system('./sienna_data')")
    print("   results = interface.run_economic_dispatch()")
    print("   interface.export_results_to_csv('./results')")