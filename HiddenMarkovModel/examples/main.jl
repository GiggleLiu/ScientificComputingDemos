using HiddenMarkovModel
using CairoMakie

"""
    create_weather_hmm()

Create a Hidden Markov Model for weather prediction with 3 states (Sunny, Cloudy, Rainy)
and 3 observations (Dry, Humid, Wet).
"""
function create_weather_hmm()
    # Transition matrix: probability of weather changing from one state to another
    A = [0.7 0.2 0.1;  # Sunny -> Sunny/Cloudy/Rainy
         0.3 0.5 0.2;  # Cloudy -> Sunny/Cloudy/Rainy
         0.2 0.4 0.4]  # Rainy -> Sunny/Cloudy/Rainy

    # Emission matrix: probability of observing conditions given the weather state
    B = [0.8 0.15 0.05;  # Sunny -> Dry/Humid/Wet
         0.2 0.6 0.2;    # Cloudy -> Dry/Humid/Wet
         0.1 0.3 0.6]    # Rainy -> Dry/Humid/Wet

    # Initial state distribution
    π = [0.4, 0.4, 0.2]  # Initial probability of Sunny/Cloudy/Rainy

    # Create the HMM
    return HMM(A, B, π)
end

"""
    print_observations(observations, observation_labels)

Print the sequence of observations with their corresponding labels.
"""
function print_observations(observations, observation_labels)
    println("\nObserved conditions:")
    for (i, obs) in enumerate(observations)
        println("Day $i: $(observation_labels[obs])")
    end
end

"""
    compare_predictions(true_states, predicted_states, state_labels, days)

Compare true states with predicted states and calculate accuracy.
"""
function compare_predictions(true_states, predicted_states, state_labels, days, title="Comparison of true weather vs. predicted weather:")
    println("\n$title")
    println("Day\tTrue Weather\tPredicted Weather\tMatch?")
    println("---\t------------\t-----------------\t------")
    correct_predictions = 0
    for i in 1:days
        match = true_states[i] == predicted_states[i]
        if match
            correct_predictions += 1
        end
        println("$i\t$(state_labels[true_states[i]])\t\t$(state_labels[predicted_states[i]])\t\t$(match ? "✓" : "✗")")
    end

    accuracy = round(correct_predictions / days * 100, digits=1)
    println("\nPrediction accuracy: $accuracy%")
    return accuracy
end

"""
    learn_hmm_from_observations(observations, n_states, n_observations, max_iter=20)

Learn HMM parameters from observations using the Baum-Welch algorithm.
"""
function learn_hmm_from_observations(observations, n_states, n_observations, max_iter=20)
    println("\nLearning HMM parameters from observations only...")
    learned_hmm = baum_welch(observations, n_states, n_observations, max_iter=max_iter)
    
    println("\nLearned transition matrix:")
    display(round.(learned_hmm.A, digits=2))
    
    println("\nLearned emission matrix:")
    display(round.(learned_hmm.B, digits=2))
    
    return learned_hmm
end

"""
    plot_results(days, true_states, predicted_states, observations, state_labels)

Create and display a plot comparing true states, predicted states, and observations.
"""
function plot_results(days, true_states, predicted_states, observations, state_labels)
    println("\nGenerating plots...")
    
    fig = Figure(size = (800, 600))
    ax = Axis(fig[1, 1], 
        xlabel = "Day", 
        ylabel = "Weather State",
        title = "Hidden Markov Model: Weather Prediction",
        yticks = (1:3, state_labels))

    # Plot true states
    lines!(ax, 1:days, true_states, label = "True Weather", color = :blue)
    scatter!(ax, 1:days, true_states, marker = :circle, color = :blue)

    # Plot predicted states
    lines!(ax, 1:days, predicted_states, label = "Predicted Weather", color = :red)
    scatter!(ax, 1:days, predicted_states, marker = :rect, color = :red)

    # Plot observations
    lines!(ax, 1:days, observations, label = "Observations", color = :green, linestyle = :dash)
    scatter!(ax, 1:days, observations, marker = :diamond, color = :green)

    # Add legend
    axislegend(ax)

    return fig
end

"""
    run_weather_example()

Run the complete weather prediction example using Hidden Markov Models.
"""
function run_weather_example()
    println("Weather Prediction Example using Hidden Markov Models")
    println("====================================================")

    # Create the weather HMM
    weather_hmm = create_weather_hmm()

    # Generate a sequence of 30 days of weather and observations
    days = 30
    println("\nGenerating a sequence of $days days of weather...")
    observations, true_states = generate_sequence(weather_hmm, days)

    # Map numeric states and observations to labels for better readability
    state_labels = ["Sunny", "Cloudy", "Rainy"]
    observation_labels = ["Dry", "Humid", "Wet"]

    # Print observations
    print_observations(observations, observation_labels)

    # Use the Viterbi algorithm to find the most likely sequence of weather states
    println("\nPredicting the most likely weather states using Viterbi algorithm...")
    predicted_states = viterbi(weather_hmm, observations)

    # Compare true states with predicted states
    accuracy = compare_predictions(true_states, predicted_states, state_labels, days)

    # Learn HMM from observations
    learned_hmm = learn_hmm_from_observations(observations, 3, 3, 20)

    # Predict states using the learned model
    learned_predictions = viterbi(learned_hmm, observations)

    # Compare predictions from the learned model
    learned_accuracy = compare_predictions(true_states, learned_predictions, state_labels, days, 
                                          "Comparison using the learned model:")

    # Plot the results
    plot_results(days, true_states, predicted_states, observations, state_labels)
    
    println("\nExample completed!")
    return weather_hmm, learned_hmm, accuracy, learned_accuracy
end

# Run the example
run_weather_example()
