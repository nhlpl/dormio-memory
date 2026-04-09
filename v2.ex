# noetic_consolidation.ex
defmodule NoeticConsolidation do
  @moduledoc """
  Sleep-wake consolidation for Noetic Memory.
  Extracts gauge-invariant facts from episodic buffer.
  """
  use GenServer
  
  defstruct [
    :episodic_buffer,     # List of %{tokens: [], phi_scores: []}
    :semantic_store,      # Persistent key-value store (e.g., ETS table)
    :consolidation_fn,    # Function to extract facts
    :phi_threshold        # Minimum Φ for a fact to be consolidated
  ]
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  # Wake phase: add conversation to episodic buffer
  def add_episode(tokens, phi_scores) do
    GenServer.cast(__MODULE__, {:add_episode, tokens, phi_scores})
  end
  
  # Sleep phase: trigger consolidation
  def consolidate do
    GenServer.call(__MODULE__, :consolidate)
  end
  
  @impl true
  def init(opts) do
    state = %__MODULE__{
      episodic_buffer: [],
      semantic_store: :ets.new(:noetic_semantic, [:set, :public, :named_table]),
      consolidation_fn: opts[:consolidation_fn] || &default_extract_facts/1,
      phi_threshold: opts[:phi_threshold] || 0.7
    }
    {:ok, state}
  end
  
  @impl true
  def handle_cast({:add_episode, tokens, phi_scores}, state) do
    episode = %{
      tokens: tokens,
      phi_scores: phi_scores,
      timestamp: DateTime.utc_now()
    }
    buffer = [episode | state.episodic_buffer] |> Enum.take(100)
    {:noreply, %{state | episodic_buffer: buffer}}
  end
  
  @impl true
  def handle_call(:consolidate, _from, state) do
    consolidated_count = 0
    
    for episode <- state.episodic_buffer do
      # Find "Noetic Monopoles" - tokens with high Φ
      high_phi_indices = episode.phi_scores
      |> Enum.with_index()
      |> Enum.filter(fn {score, _} -> score >= state.phi_threshold end)
      |> Enum.map(&elem(&1, 1))
      
      if high_phi_indices != [] do
        # Extract gauge-invariant facts around high-Φ tokens
        facts = extract_facts_around(episode.tokens, high_phi_indices)
        
        for fact <- facts do
          # Store in semantic memory with content-addressable key
          key = hash_fact(fact)
          :ets.insert(:noetic_semantic, {key, fact})
          consolidated_count = consolidated_count + 1
        end
      end
    end
    
    # Clear episodic buffer after consolidation (graduated dissolution)
    {:reply, consolidated_count, %{state | episodic_buffer: []}}
  end
  
  defp extract_facts_around(tokens, high_phi_indices) do
    # Simplified: extract subject-predicate-object triples
    # In production, use an LLM to extract structured knowledge
    for idx <- high_phi_indices do
      context = Enum.slice(tokens, max(0, idx-3), min(length(tokens)-idx+3, 7))
      "Fact: #{Enum.join(context, " ")}"
    end
  end
  
  defp hash_fact(fact) do
    :crypto.hash(:sha256, fact) |> Base.encode16()
  end
  
  defp default_extract_facts(_), do: []
end
