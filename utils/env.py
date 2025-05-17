import gym
import numpy as np
import osmnx as ox
import networkx as nx
from gym import spaces
from math import radians, sin, cos, sqrt, atan2

class EVRouteEnv(gym.Env):
    def __init__(self, G, spklu_df, battery_capacity_kwh,
                 start, goal, connector_type, initial_soc, max_steps=200):
        super().__init__()
        self.G         = G
        self.spklu     = spklu_df
        self.capacity  = battery_capacity_kwh
        self.initial_soc = initial_soc if initial_soc is not None else battery_capacity_kwh
        self.connector = connector_type
        self.min_soc   = 0.2 * self.capacity
        self.max_steps = max_steps

        # map start/goal → nearest node
        self.start = ox.distance.nearest_nodes(G, *ox.geocode(start)[::-1])
        self.goal  = ox.distance.nearest_nodes(G, *ox.geocode(goal)[::-1])

        # action_space = max degree + 1 (for “charge”)
        self.max_deg      = max(dict(self.G.degree()).values())
        self.action_space = spaces.Discrete(self.max_deg + 2)

        # observation_space = [dist_norm, soc_ratio, norm_spklu, step_ratio]
        K = 3
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4 + 3*K,), dtype=np.float32
        )

        # precompute max possible distance (diagonal bounding box)
        ys = [d['y'] for _,d in G.nodes(data=True)]
        xs = [d['x'] for _,d in G.nodes(data=True)]
        self._max_dist = self._haversine_coords(min(ys), min(xs), max(ys), max(xs))

        # fallback consumption (kWh per km)
        self.fallback_consumption_kwh_per_km = 0.15

        # reward scalars
        self.step_penalty    = -0.1
        self.low_soc_penalty = -10.0
        self.charge_reward   = +20.0
        self.goal_reward     = +100.0
        self.death_penalty   = -50.0
        self.recommendation_reward = +10.0
        self.distance_to_goal_penalty = -10.0

        # initialize dynamic vars
        self.reset()

    def reset(self):
        self.node          = self.start
        self.soc           = self.initial_soc
        self.visited_spklu = []
        self.done          = False
        self.step_count    = 0
        # initialize path tracking
        self.path = [self.node]
        self.trajectory = [self.node]

        # reverse graph
        self.reversed_G = self.G.reverse(copy=False)

        self.dist_to_goal = {}
        self.next_hop_to_goal = {}

        self.compute_path_to_goal()

        # SPKLU
        self.dist_to_spklu = {}
        self.next_hop_to_spklu = {}

        if self.soc < self.dist_to_goal.get(self.start, float('inf')):
            self.compute_path_to_spklu()

        self._update_action_list()
        return self._get_state()

    def step(self, action_idx):
        actual = self.action_list[action_idx]

        reward = 0.0
        prev_dist = self._haversine_distance(self.node, self.goal)

        if actual == 'stay':
            self.done = True
            reward += self.death_penalty

        elif actual == 'charge':
            # Charge action
            if self.G.nodes[self.node].get('is_spklu') and self.connector in self.G.nodes[self.node].get('available_connectors', []):
                self.soc = self.capacity
                if self.node not in self.visited_spklu:
                    spklu_name = self.G.nodes[self.node].get('spklu_name')
                    self.visited_spklu.append({'spklu_name': spklu_name, 'node': self.node})

                reward += self.charge_reward
            else:
                reward += self.death_penalty

        else:
            # stay-put if no valid edge
            if not self.G.has_edge(self.node, actual):
                reward += self.step_penalty
            else:
                edge_data = self.G.get_edge_data(self.node, actual)
                attrs = edge_data[list(edge_data.keys())[0]]

                # distance and energy
                distance_km = attrs.get('distance', attrs.get('length',0)/1000)
                energy = attrs.get('energy', distance_km * self.fallback_consumption_kwh_per_km)

                # update SOC & node
                self.soc  -= energy
                self.node = actual

                # reward shaping
                reward += self.step_penalty
                if self.soc < self.min_soc:
                    reward += self.low_soc_penalty

                new_dist = self._haversine_distance(self.node, self.goal)
                reward += (prev_dist - new_dist) * 50

                if actual == self.action_list[0]:
                    reward += self.recommendation_reward

                # jarak ke goal bertambah
                delta = new_dist - prev_dist
                if delta > 0:
                    reward += self.distance_to_goal_penalty

                if self.soc <= 0:
                    self.done = True
                    reward += self.death_penalty

        # rekam setiap pindah (termasuk revisit)
        if actual != 'charge':
            self.trajectory.append(self.node)

        # goal check
        if self.node == self.goal:
            self.done = True
            reward += self.goal_reward

        # step limit
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        # track path (only when moving to a new node)
        if actual != 'charge' and self.node not in self.path:
            self.path.append(self.node)

        # update actions for next state
        self._update_action_list()
        return self._get_state(), reward, self.done, {
            'visited_spklu': self.visited_spklu,
            'path': list(self.path)
        }

    def _update_action_list(self):
        def insert_front(item, lst):
            if item in lst:
                lst.remove(item)
            lst.insert(0, item)

        # Ambil semua neighbor dari node sekarang
        neighbors = list(self.G.neighbors(self.node)) or [self.node]

        # Ambil neighbor yang belum dikunjungi
        acts = [n for n in neighbors if n not in self.path] or neighbors.copy()
        needed_goal = self.dist_to_goal.get(self.node, float('inf'))

        # Tambahkan opsi "charge" jika ada SPKLU dan energi kurang dari jarak ke goal
        if( self.G.nodes[self.node].get('is_spklu') and
            self.soc < needed_goal and
            self.connector in self.G.nodes[self.node].get('available_connectors', [])
          ):
            acts.append('charge')

        # Jika daftar aksi kosong, tetap di tempat
        if not acts:
            acts = [self.node]

        # Rekomendasikan neighbor terbaik
        best = self.recommend_neighbor()
        insert_front(best, acts)

        # Jika cukup energi ke goal
        if self.soc >= needed_goal:
            hop = self.next_hop_to_goal.get(self.node)
            insert_front(hop, acts)
        else:
            # Tidak cukup energi ke goal, coba ke SPKLU
            cost_spk = self.dist_to_spklu.get(self.node, float('inf'))
            if cost_spk < float('inf') and self.soc >= cost_spk:
                hop_spk = self.next_hop_to_spklu.get(self.node)
                insert_front(hop_spk, acts)

                is_spk_here = self.G.nodes[self.node].get('is_spklu')
                available_conns = self.G.nodes[self.node].get('available_connectors', [])

                if is_spk_here and self.soc < needed_goal and self.connector in available_conns:
                    insert_front('charge', acts)
            else:
                insert_front('stay', acts)

        # Tambahkan duplikasi aksi hingga sepanjang action_space.n
        while len(acts) < self.action_space.n:
            acts.append(acts[0])

        self.action_list = acts[: self.action_space.n]

    def recommend_neighbor(self, alpha=0.7, beta=0.3):
        neighbors = list(self.G.neighbors(self.node))
        best, best_score = None, float('inf')
        for n in neighbors:
            # jarak ke goal
            d_goal = self._haversine_distance(n, self.goal) / self._max_dist
            # energy cost ke neighbor
            energy = self.G[self.node][n].get('energy',
                      self.G[self.node][n].get(
                          'distance',
                          self.G[self.node][n].get('length', 0) / 1000
                      ) * self.fallback_consumption_kwh_per_km)
            # normalisasi energy (asumsi E_max = capacity)
            e_norm = energy / self.capacity
            score = alpha * d_goal + beta * e_norm
            if score < best_score:
                best_score, best = score, n
        return best

    def _get_state(self):
        # normalized distance to goal
        dist = self._haversine_distance(self.node, self.goal)
        dist_norm = dist / self._max_dist

        # state of charge ratio
        soc_ratio = max(0.0, self.soc) / self.capacity

        # normalized distance to nearest SPKLU
        spklu_nodes = [n for n, d in self.G.nodes(data=True) if d.get('is_spklu')]
        if spklu_nodes:
            dists_spklu = [self._haversine_distance(self.node, n) for n in spklu_nodes]
            norm_spklu = min(dists_spklu) / self._max_dist
        else:
            norm_spklu = 1.0

        # remaining step ratio
        step_ratio = (self.max_steps - self.step_count) / self.max_steps

        # ** Add neighbors info to the state **
        K = 3
        nbrs = list(self.G.neighbors(self.node))[:K]

        neighbor_info = []
        for neighbor in nbrs:
            d_goal = self._haversine_distance(neighbor, self.goal) / self._max_dist
            d_step = self.G[self.node][neighbor].get(
                'distance',
                self.G[self.node][neighbor].get('length', 0) / 1000
            )
            e_cost = self.G[self.node][neighbor].get('energy', 0)
            neighbor_info.extend([d_goal, d_step, e_cost])

        # pad kalau kurang dari K
        while len(neighbor_info) < 3 * K:
            neighbor_info.append(0.0)

        state = np.array(
            [dist_norm, soc_ratio, norm_spklu, step_ratio] + neighbor_info,
            dtype=np.float32
        )
        return state

    def _get_node_coords(self, node):
        d = self.G.nodes[node]
        return d['y'], d['x']

    def _haversine_distance(self, a, b):
        return self._haversine_coords(*self._get_node_coords(a),
                                      *self._get_node_coords(b))

    def _haversine_coords(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def compute_path_to_goal(self):
        dist_all, paths_all = nx.single_source_dijkstra(
            self.reversed_G, source=self.goal, weight='energy'
        )
        # simpan energi
        self.dist_to_goal = dist_all
        # build next_hop: dari node n → goal
        self.next_hop_to_goal = {}
        for node, seq in paths_all.items():
            rev = seq[::-1]  # [node, ..., goal]
            # jika panjang >1, next hop = elemen ke-1, else node == goal
            self.next_hop_to_goal[node] = rev[1] if len(rev) > 1 else node

    def compute_path_to_spklu(self):
        spk_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get('is_spklu') and self.connector in d.get('available_connectors', [])
        ]
        if not spk_nodes:
            self.dist_to_spklu = {}
            self.next_hop_to_spklu = {}
            return

        dist_all, paths_all = nx.multi_source_dijkstra(
            self.reversed_G, sources=spk_nodes, weight='energy'
        )

        # Simpan jarak ke SPKLU terdekat untuk setiap node
        self.dist_to_spklu = dist_all

        # Bangun next_hop: dari node → SPKLU
        self.next_hop_to_spklu = {}
        for node, seq in paths_all.items():
            rev = seq[::-1]
            self.next_hop_to_spklu[node] = rev[1] if len(rev) > 1 else node