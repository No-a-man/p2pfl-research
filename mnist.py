import argparse
import time
import uuid
import numpy as np

import matplotlib.pyplot as plt
import runSimpleTextSVM

from p2pfl.communication.protocols.protobuff.grpc import GrpcCommunicationProtocol
from p2pfl.communication.protocols.protobuff.memory import MemoryCommunicationProtocol
from p2pfl.learning.aggregators.scaffold import Scaffold
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from datasets import load_dataset
from p2pfl.learning.dataset.partition_strategies import RandomIIDPartitionStrategy
from p2pfl.management.logger import logger
from p2pfl.node import Node
from p2pfl.settings import Settings
from p2pfl.utils.topologies import TopologyFactory, TopologyType
from p2pfl.utils.utils import set_standalone_settings, wait_convergence, wait_to_finish


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P2PFL MNIST experiment using the Web Logger.")
    parser.add_argument("--nodes", type=int, help="The number of nodes.", default=2)
    parser.add_argument("--rounds", type=int, help="The number of rounds.", default=2)
    parser.add_argument("--epochs", type=int, help="The number of epochs.", default=1)
    parser.add_argument("--show_metrics", action="store_true", help="Show metrics.", default=True)
    parser.add_argument("--measure_time", action="store_true", help="Measure time.", default=False)
    parser.add_argument("--token", type=str, help="The API token for the Web Logger.", default="")
    parser.add_argument("--protocol", type=str, help="The protocol to use.", default="grpc", choices=["grpc", "unix", "memory"])
    parser.add_argument("--framework", type=str, help="The framework to use.", default="pytorch", choices=["pytorch", "tensorflow", "flax"])
    parser.add_argument(
        "--model",
        type=str,
        help="Which model to use when framework is pytorch.",
        default="simpletext",
        choices=["simpletext", "binary"],
    )
    parser.add_argument("--aggregator", type=str, help="The aggregator to use.", default="fedavg", choices=["fedavg", "scaffold"])
    parser.add_argument("--profiling", action="store_true", help="Enable profiling.", default=False)
    parser.add_argument("--reduced_dataset", action="store_true", help="Use a reduced dataset just for testing.", default=False)
    parser.add_argument("--use_scaffold", action="store_true", help="Use the Scaffold aggregator.", default=False)
    parser.add_argument("--seed", type=int, help="The seed to use.", default=666)
    parser.add_argument("--batch_size", type=int, help="The batch size for training.", default=128)
    parser.add_argument(
        "--topology",
        type=str,
        choices=[t.value for t in TopologyType],
        default="line",
        help="The network topology (star, full, line, ring).",
    )
    args = parser.parse_args()
    # parse topology to TopologyType enum
    args.topology = TopologyType(args.topology)

    return args


def mnist(
    n: int,
    r: int,
    e: int,
    show_metrics: bool = True,
    measure_time: bool = False,
    protocol: str = "grpc",
    framework: str = "pytorch",
    aggregator: str = "fedavg",
    reduced_dataset: bool = False,
    topology: TopologyType = TopologyType.LINE,
    batch_size: int = 32,
    model_type: str = "simpletext",
) -> None:
    """
    P2PFL MNIST experiment.

    Args:
        n: The number of nodes.
        r: The number of rounds.
        e: The number of epochs.
        show_metrics: Show metrics.
        measure_time: Measure time.
        protocol: The protocol to use.
        framework: The framework to use.
        aggregator: The aggregator to use.
        reduced_dataset: Use a reduced dataset just for testing.
        topology: The network topology (star, full, line, ring).
        batch_size: The batch size for training.

    """
    # Always measure time for training
    training_start_time = time.time()
    print(f"🚀 Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check settings
    if n > Settings.gossip.TTL:
        raise ValueError(
            "For in-line topology TTL must be greater than the number of nodes." "Otherwise, some messages will not be delivered."
        )

    # Imports
    if framework == "tensorflow":
        from p2pfl.examples.mnist.model.mlp_tensorflow import model_build_fn  # type: ignore

        model_fn = model_build_fn  # type: ignore
    elif framework == "pytorch":
        # Choose between the existing SimpleTextSVM (text-based) and the Binary SVM
        if model_type == "binary":
            from runBinarySVM import model_build_fn  # type: ignore
        else:
            from runSimpleTextSVM import model_build_fn  # type: ignore

        model_fn = model_build_fn  # type: ignore
    else:
        raise ValueError(f"Framework {args.framework} not added on this example.")

    # Data - Use original MNIST dataset for now
    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")
    data.set_batch_size(batch_size)
    partitions = data.generate_partitions(
        n * 50 if reduced_dataset else n,
        RandomIIDPartitionStrategy,  # type: ignore
    )

    # Node Creation
    nodes = []
    for i in range(n):
        address = f"node-{i}" if protocol == "memory" else f"unix:///tmp/p2pfl-{i}.sock" if protocol == "unix" else "127.0.0.1"

        # Nodes
        node = Node(
            model_fn(),
            partitions[i],
            protocol=MemoryCommunicationProtocol() if protocol == "memory" else GrpcCommunicationProtocol(),
            addr=address,
            aggregator=Scaffold() if aggregator == "scaffold" else None,
        )
        node.start()
        nodes.append(node)

    try:
        adjacency_matrix = TopologyFactory.generate_matrix(topology, len(nodes))
        TopologyFactory.connect_nodes(adjacency_matrix, nodes)

        wait_convergence(nodes, n - 1, only_direct=False, wait=60)  # type: ignore

        if r < 1:
            raise ValueError("Skipping training, amount of round is less than 1")

        # Start Learning
        learning_start_time = time.time()
        print(f"📚 Starting federated learning at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔢 Configuration: {r} rounds, {e} epochs per round, {n} nodes")
        nodes[0].set_start_learning(rounds=r, epochs=e)

        # Wait and check
        wait_to_finish(nodes, timeout=60 * 60)  # 1 hour
        
        # Calculate training time
        learning_end_time = time.time()
        total_training_time = learning_end_time - learning_start_time
        total_experiment_time = learning_end_time - training_start_time
        
        print(f"\n✅ Training completed!")
        print(f"⏱️ 	Learning time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        print(f"⏱️ 	Average time per round: {total_training_time/r:.2f} seconds")
        print(f"⏱️ 	Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")
        print(f"🏁 Experiment finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Calculate Average Metrics Across All Nodes
        if show_metrics:
            print(f"\n📊 CALCULATING AVERAGE METRICS ACROSS ALL NODES:")
            print(f"=" * 60)
            
            # Get all local logs from all nodes
            local_logs = logger.get_local_logs()
            if local_logs != {}:
                all_accuracies = []
                all_losses = []
                # collect per-node loss histories for epoch-vs-loss plotting
                loss_histories = []
                
                # Debug: Print available metrics for ALL nodes
                print(f"🔍 DEBUG: Available metrics in logs:")
                total_nodes_found = 0
                for node_id, node_logs in local_logs.items():
                    for round_num, round_metrics in node_logs.items():
                        for node_name, node_metrics in round_metrics.items():
                            total_nodes_found += 1
                            print(f" 	Node {node_name} metrics: {list(node_metrics.keys())}")
                            # Check if this node has test results
                            has_test_acc = any(metric in node_metrics for metric in ['test_acc', 'binary_accuracy', 'train_binary_acc'])
                            print(f" 	 	Has test accuracy: {has_test_acc}")
                        break  # Just show first round for debugging
                    break  # Just show first round for debugging
                
                print(f"📊 Total nodes found in logs: {total_nodes_found}")
                print(f"📊 Expected nodes: {n}")
                if total_nodes_found < n:
                    print(f"⚠️ 	WARNING: Only {total_nodes_found} out of {n} nodes have logged metrics!")
                    print(f" 	 This could be due to:")
                    print(f" 	 - Some nodes not completing evaluation")
                    print(f" 	 - Logging issues in some nodes")
                    print(f" 	 - Network communication problems")
                
                # Collect metrics from all nodes
                for node_id, node_logs in local_logs.items():
                    for round_num, round_metrics in node_logs.items():
                        for node_name, node_metrics in round_metrics.items():
                            # Get final accuracy and loss for this node
                            # Check multiple possible metric names
                            final_acc = None
                            final_loss = None
                            
                            # Try different accuracy metric names
                            for acc_metric in ['test_acc', 'binary_accuracy', 'train_binary_acc']:
                                if acc_metric in node_metrics and node_metrics[acc_metric]:
                                    final_acc = node_metrics[acc_metric][-1][1]  # Get last accuracy value
                                    break
                            
                            # Try different loss metric names
                            for loss_metric in ['loss', 'train_loss']:
                                if loss_metric in node_metrics and node_metrics[loss_metric]:
                                    final_loss = node_metrics[loss_metric][-1][1]  # Get last loss value
                                    break
                            
                            if final_acc is not None:
                                all_accuracies.append(final_acc)
                                print(f"Node {node_name}: {final_acc:.4f} ({final_acc*100:.2f}%)")
                            
                            if final_loss is not None:
                                all_losses.append(final_loss)
                            # collect full loss history series (if available) for plotting
                            for loss_metric in ['loss', 'train_loss']:
                                if loss_metric in node_metrics and node_metrics[loss_metric]:
                                    try:
                                        # node_metrics[loss_metric] is expected to be a list of (step, value) pairs
                                        series = [v for (_, v) in node_metrics[loss_metric]]
                                        if len(series) > 0:
                                            loss_histories.append(series)
                                            break
                                    except Exception:
                                        # ignore malformed entries
                                        pass
                            
                # Calculate averages
                if all_accuracies:
                    avg_accuracy = sum(all_accuracies) / len(all_accuracies)
                    print(f"\n🎯 AVERAGE ACCURACY ACROSS ALL NODES: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
                    print(f"📈 Accuracy Range: {min(all_accuracies):.4f} - {max(all_accuracies):.4f}")
                    print(f"📊 Standard Deviation: {np.std(all_accuracies):.4f}")
                
                if all_losses:
                    avg_loss = sum(all_losses) / len(all_losses)
                    print(f"📉 AVERAGE LOSS ACROSS ALL NODES: {avg_loss:.4f}")
                    print(f"📈 Loss Range: {min(all_losses):.4f} - {max(all_losses):.4f}")
                
                print(f"🔢 Total Nodes Evaluated: {len(all_accuracies)}")
                print(f"=" * 60)
                
                # If we have fewer metrics than expected nodes, try global logs
                if len(all_accuracies) < n:
                    print(f"⚠️ 	Only {len(all_accuracies)} nodes have metrics, trying global logs for missing nodes...")
                    global_logs = logger.get_global_logs()
                    if global_logs != {}:
                        print(f"🔍 DEBUG: Available global metrics:")
                        global_nodes_found = 0
                        for node_name, node_metrics in global_logs.items():
                            global_nodes_found += 1
                            print(f" 	Global {node_name} metrics: {list(node_metrics.keys())}")

                            # Try to get metrics from global logs
                            for acc_metric in ['test_acc', 'binary_accuracy', 'train_binary_acc']:
                                if acc_metric in node_metrics and node_metrics[acc_metric]:
                                    final_acc = node_metrics[acc_metric][-1][1]
                                    all_accuracies.append(final_acc)
                                    print(f"Global Node {node_name}: {final_acc:.4f} ({final_acc*100:.2f}%)")
                                    break

                            for loss_metric in ['loss', 'train_loss']:
                                if loss_metric in node_metrics and node_metrics[loss_metric]:
                                    final_loss = node_metrics[loss_metric][-1][1]
                                    all_losses.append(final_loss)
                                    # try to capture full series for plotting
                                    try:
                                        series = [v for (_, v) in node_metrics[loss_metric]]
                                        if len(series) > 0:
                                            loss_histories.append(series)
                                    except Exception:
                                        pass
                                    break
                        
                        print(f"📊 Global nodes found: {global_nodes_found}")
                        print(f"📊 Total nodes with metrics: {len(all_accuracies)}")
                        
                        if len(all_accuracies) < n:
                            print(f"⚠️ 	Still missing {n - len(all_accuracies)} nodes!")
                            print(f" 	 Possible reasons:")
                            print(f" 	 - Some nodes failed to complete training")
                            print(f" 	 - Some nodes had evaluation errors")
                            print(f" 	 - Network timeouts or communication issues")
                            print(f" 	 - Insufficient data for some nodes")
                        
                        # Recalculate averages if we found metrics in global logs
                        if all_accuracies:
                            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
                            print(f"\n🎯 AVERAGE ACCURACY FROM GLOBAL LOGS: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
                            print(f"📈 Accuracy Range: {min(all_accuracies):.4f} - {max(all_accuracies):.4f}")
                            print(f"📊 Standard Deviation: {np.std(all_accuracies):.4f}")
                            
                            if all_losses:
                                avg_loss = sum(all_losses) / len(all_losses)
                                print(f"📉 AVERAGE LOSS FROM GLOBAL LOGS: {avg_loss:.4f}")
                                print(f"📈 Loss Range: {min(all_losses):.4f} - {max(all_losses):.4f}")
                            
                            print(f"🔢 Total Nodes Evaluated: {len(all_accuracies)}")
                            print(f"=" * 60)

                # --- Epoch vs Loss plot (show first) ---
                try:
                    print(f"DEBUG: collected {len(loss_histories)} loss series and {len(all_losses)} final losses")
                except Exception:
                    print("DEBUG: could not introspect loss_histories/all_losses")

                if loss_histories:
                    try:
                        # Use the requested epoch count (e) as the x-axis length when available
                        target_epochs = e if (isinstance(e, int) and e > 0) else max(len(s) for s in loss_histories)
                        max_len = max(len(s) for s in loss_histories)
                        arr = np.full((len(loss_histories), target_epochs), np.nan, dtype=float)
                        for i, s in enumerate(loss_histories):
                            copy_len = min(len(s), target_epochs)
                            if copy_len > 0:
                                arr[i, :copy_len] = s[:copy_len]

                        mean_loss = np.nanmean(arr, axis=0)
                        std_loss = np.nanstd(arr, axis=0)
                        epochs = np.arange(1, target_epochs + 1)

                        plt.figure(figsize=(8, 5))
                        # Plot per-node series faintly to show variance (trim/pad to target_epochs)
                        for s in loss_histories:
                            ep = np.arange(1, min(len(s), target_epochs) + 1)
                            plt.plot(ep, s[: len(ep)], color='gray', alpha=0.25, linewidth=1)

                        # Plot mean and std band across target_epochs
                        plt.plot(epochs, mean_loss, marker='o', color='red', label='Mean Loss')
                        plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, color='red', alpha=0.2, label='Std')
                        plt.xlabel('Epoch')
                        plt.ylabel('Loss')
                        plt.title(f'Epoch vs Loss (averaged across nodes) - {target_epochs} epochs')
                        plt.xlim(1, target_epochs)
                        plt.grid(True)
                        plt.legend()
                        plt.tight_layout()
                        try:
                            plt.savefig('epoch_vs_loss_across_nodes.png')
                            print('📈 Saved epoch vs loss plot to epoch_vs_loss_across_nodes.png')
                        except Exception:
                            pass
                        plt.show()
                    except Exception as e:
                        print(f"⚠️ Could not create epoch-vs-loss plot: {e}")
                else:
                    # Fallback: if we don't have full series but have final losses, plot those
                    if all_losses:
                        try:
                            plt.figure(figsize=(8, 5))
                            x = np.arange(1, len(all_losses) + 1)
                            plt.bar(x, all_losses, color='skyblue', alpha=0.8)
                            plt.xlabel('Node')
                            plt.ylabel('Final Loss')
                            plt.title('Final Loss per Node (fallback)')
                            plt.grid(True, axis='y', alpha=0.3)
                            plt.tight_layout()
                            plt.savefig('final_loss_per_node.png')
                            print('📈 Saved fallback final per-node loss plot to final_loss_per_node.png')
                            plt.show()
                        except Exception as e:
                            print(f"⚠️ Could not create fallback final-loss plot: {e}")

                # Create a single graph showing average accuracy with standard deviation
                if all_accuracies:
                    plt.figure(figsize=(10, 6))
                    
                    # Create bar chart for average accuracy
                    x_pos = [0]
                    y_pos = [avg_accuracy]
                    y_err = [np.std(all_accuracies)]
                    
                    plt.bar(x_pos, y_pos, yerr=y_err, capsize=10, 
                            color='skyblue', alpha=0.7, width=0.5,
                            label=f'Average Accuracy: {avg_accuracy:.4f} ± {np.std(all_accuracies):.4f}')
                    
                    # Add individual node points - use simple positions
                    node_positions = np.arange(len(all_accuracies))
                    plt.scatter(node_positions, all_accuracies, color='red', alpha=0.6, s=50, 
                                 label=f'Individual Nodes (n={len(all_accuracies)})')
                    
                    # Customize the plot
                    plt.title('Federated Learning: Average Accuracy Across All Nodes', fontsize=14, fontweight='bold')
                    plt.xlabel('Nodes', fontsize=12)
                    plt.ylabel('Accuracy', fontsize=12)
                    plt.ylim(0, max(all_accuracies) * 1.2)  # Set y-axis range
                    
                    # Add text annotations
                    plt.text(0, avg_accuracy + np.std(all_accuracies) + 0.01, 
                             f'Avg: {avg_accuracy:.4f}\nStd: {np.std(all_accuracies):.4f}', 
                             ha='center', va='bottom', fontsize=10, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
                    
                    # Add grid
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.show()
                    
                    print(f"\n📊 Graph created: Average Accuracy with Standard Deviation")
                    print(f" 	 - Average: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
                    print(f" 	 - Standard Deviation: {np.std(all_accuracies):.4f}")
                    print(f" 	 - Range: {min(all_accuracies):.4f} - {max(all_accuracies):.4f}")

                
                # (moved) epoch-vs-loss plot previously here has been displayed earlier
    except Exception as e:
        raise e
    finally:
        # Stop Nodes
        for node in nodes:
            node.stop()

        # Final timing summary
        final_time = time.time()
        total_experiment_time = final_time - training_start_time
        print(f"\n📊 FINAL TIMING SUMMARY:")
        print(f"⏱️ 	Total experiment time: {total_experiment_time:.2f} seconds ({total_experiment_time/60:.2f} minutes)")
        print(f"🏁 Complete experiment finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if measure_time:
            print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Parse args
    args = __parse_args()

    set_standalone_settings()

    if args.profiling:
        import os  # noqa: I001
        import yappi  # type: ignore

        # Start profiler
        yappi.start()

    # Set logger
    if args.token != "":
        logger.connect_web("http://localhost:3000/api/v1", args.token)

    # Seed
    if args.seed is not None:
        Settings.general.SEED = args.seed

    # Launch experiment
    try:
        mnist(
            args.nodes,
            args.rounds,
            args.epochs,
            show_metrics=args.show_metrics,
            measure_time=args.measure_time,
            protocol=args.protocol,
            framework=args.framework,
            aggregator=args.aggregator,
            reduced_dataset=args.reduced_dataset,
            topology=args.topology,
            batch_size=args.batch_size,
            model_type=args.model,
        )
    finally:
        if args.profiling:
            # Stop profiler
            yappi.stop()
            # Save stats
            profile_dir = os.path.join("profile", "mnist", str(uuid.uuid4()))
            os.makedirs(profile_dir, exist_ok=True)
            for thread in yappi.get_thread_stats():
                yappi.get_func_stats(ctx_id=thread.id).save(f"{profile_dir}/{thread.name}-{thread.id}.pstat", type="pstat")