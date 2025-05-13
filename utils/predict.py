from utils.agent  import DQNAgent
from utils.env import EVRouteEnv
import numpy as np

def predict(
        G, model, df_spklu, soc_pct, capacity_kwh, start_address, goal_address,
        connector, max_steps
):
    initial_soc = soc_pct / 100 * capacity_kwh
    env = EVRouteEnv(
        G, df_spklu,
        battery_capacity_kwh=capacity_kwh,
        start=start_address,
        goal=goal_address,
        connector_type=connector,
        initial_soc=initial_soc,
        max_steps=max_steps
    )

    # validasi energy
    needed_spk, spk_node = env.estimate_energi_to_nearest_spklu()
    if spk_node is None:
        return {'stop': True}

    if initial_soc < needed_spk:
        return {'stop': True}

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    agent.model = model
    agent.epsilon = 0.1

    state = env.reset()
    done = False
    total_distance = 0.0
    total_energy = 0.0

    while not done:
        q_values = agent.model.predict(np.expand_dims(state, axis=0), verbose=0)
        action_idx = np.argmax(q_values[0])
        state, reward, done, info = env.step(action_idx)

        actual = env.action_list[action_idx]
        if actual != 'charge':
            u, v = env.trajectory[-2], env.trajectory[-1]
            if u != v and env.G.has_edge(u, v):
                attrs = env.G.get_edge_data(u, v)[0]
                dist_km = attrs.get('distance', attrs.get('length', 0) / 1000)
                energy = attrs.get('energy', dist_km * env.fallback_consumption_kwh_per_km)
                total_distance += dist_km
                total_energy += energy

    path_nodes = info.get('path', [])
    route_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path_nodes]
    enough_energy = capacity_kwh >= total_energy
    status = "✅ Enough" if enough_energy else "❌ Not Enough"

    return {
        'stop': False,
        'route_coords': route_coords,
        'info': info,
        'path_nodes': path_nodes,
        'total_distance': total_distance,
        'total_energy': total_energy,
        'status': status,
        'capacity_kwh': capacity_kwh,
        'soc_pct': soc_pct
    }