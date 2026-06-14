"""Tests Phase 8 — Oracle trajectoire enrichi (add_control_labels_v2).

Vérifie que l'oracle v2 :
  - donne un RPM plus élevé en montée rapide qu'en refroidissement à même T
  - force RPM_MAX en cas d'urgence panne imminente
  - retourne action_class_v2=0 pour une machine froide sans trajectoire notable
  - ne régresse pas sur add_control_labels (oracle v1)
  - est cohérent avec RPM_LEVELS [800, 1500, 2500, 3500, 4500]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.labeler import (
    RPM_LEVELS,
    add_control_labels,
    add_control_labels_v2,
    label_names_control,
    label_names_control_v2,
)

T_SHUTDOWN = 88.0
N_LEVELS = len(RPM_LEVELS)  # 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(
    temperature_c: float,
    temp_delta_30s: float = 0.0,
    time_to_failure_s: float | None = None,
    status: str = "on",
    n: int = 1,
) -> pd.DataFrame:
    """Construit un DataFrame minimal pour les tests."""
    data = {
        "temperature_c": [temperature_c] * n,
        "temp_delta_30s": [temp_delta_30s] * n,
        "status": [status] * n,
    }
    if time_to_failure_s is not None:
        data["time_to_failure_s"] = [time_to_failure_s] * n
    else:
        data["time_to_failure_s"] = [np.nan] * n
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests oracle v2 : comportement directionnel
# ---------------------------------------------------------------------------

class TestOracleV2Directionnel:

    def test_montee_rapide_rpm_superieur_au_refroidissement_meme_temperature(self):
        """A T identique, une montée rapide doit donner un RPM >= refroidissement."""
        T = 70.0
        df_montee = make_df(T, temp_delta_30s=+4.0)
        df_descente = make_df(T, temp_delta_30s=-4.0)

        df_montee = add_control_labels_v2(df_montee, t_shutdown_c=T_SHUTDOWN)
        df_descente = add_control_labels_v2(df_descente, t_shutdown_c=T_SHUTDOWN)

        assert df_montee["action_class_v2"].iloc[0] > df_descente["action_class_v2"].iloc[0], (
            f"Montée +4°C/30s devrait donner RPM > descente -4°C/30s à T={T}°C. "
            f"Obtenu montee={df_montee['action_class_v2'].iloc[0]}, "
            f"descente={df_descente['action_class_v2'].iloc[0]}"
        )

    def test_montee_moderee_rpm_superieur_ou_egal_stabilite(self):
        """Montée modérée doit donner RPM >= stabilité à même T."""
        T = 65.0
        df_montee = make_df(T, temp_delta_30s=+2.0)
        df_stable = make_df(T, temp_delta_30s=0.0)

        df_montee = add_control_labels_v2(df_montee, t_shutdown_c=T_SHUTDOWN)
        df_stable = add_control_labels_v2(df_stable, t_shutdown_c=T_SHUTDOWN)

        assert df_montee["action_class_v2"].iloc[0] >= df_stable["action_class_v2"].iloc[0]

    def test_refroidissement_rpm_inferieur_ou_egal_stabilite(self):
        """Refroidissement doit donner RPM <= stabilité (beta contribue 0 si delta<0)."""
        T = 70.0
        df_descente = make_df(T, temp_delta_30s=-3.0)
        df_stable = make_df(T, temp_delta_30s=0.0)

        df_descente = add_control_labels_v2(df_descente, t_shutdown_c=T_SHUTDOWN)
        df_stable = add_control_labels_v2(df_stable, t_shutdown_c=T_SHUTDOWN)

        assert df_descente["action_class_v2"].iloc[0] <= df_stable["action_class_v2"].iloc[0]


# ---------------------------------------------------------------------------
# Tests oracle v2 : urgence panne
# ---------------------------------------------------------------------------

class TestOracleV2Urgence:

    def test_panne_imminente_force_rpm_max(self):
        """time_to_failure_s très court doit forcer action_class_v2 = n_levels-1."""
        # Machine froide mais panne dans 5s (< horizon/3 = 20s)
        df = make_df(temperature_c=45.0, temp_delta_30s=0.0, time_to_failure_s=5.0)
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN, horizon_s=60.0)
        assert df["action_class_v2"].iloc[0] == N_LEVELS - 1, (
            f"Panne dans 5s devrait forcer action_class_v2={N_LEVELS-1}, "
            f"obtenu {df['action_class_v2'].iloc[0]}"
        )

    def test_panne_lointaine_faible_urgence(self):
        """time_to_failure_s >> horizon ne doit pas augmenter significativement le score."""
        df_sans_panne = make_df(temperature_c=60.0, temp_delta_30s=0.0)
        df_panne_lointaine = make_df(temperature_c=60.0, temp_delta_30s=0.0,
                                     time_to_failure_s=300.0)

        df_sans_panne = add_control_labels_v2(df_sans_panne, t_shutdown_c=T_SHUTDOWN,
                                               horizon_s=60.0)
        df_panne_lointaine = add_control_labels_v2(df_panne_lointaine,
                                                    t_shutdown_c=T_SHUTDOWN, horizon_s=60.0)

        # urgency = clip(1 - 300/60, 0, 1) = 0 -> même résultat
        assert df_sans_panne["action_class_v2"].iloc[0] == \
               df_panne_lointaine["action_class_v2"].iloc[0]

    def test_panne_horizon_moyen_urgence_partielle(self):
        """Panne dans 30s (= 50% de horizon=60s) -> urgency=0.5 -> classe augmente."""
        df_sans_urgence = make_df(temperature_c=55.0, temp_delta_30s=0.0)
        df_urgence = make_df(temperature_c=55.0, temp_delta_30s=0.0,
                             time_to_failure_s=30.0)

        df_sans_urgence = add_control_labels_v2(df_sans_urgence, t_shutdown_c=T_SHUTDOWN)
        df_urgence = add_control_labels_v2(df_urgence, t_shutdown_c=T_SHUTDOWN,
                                           horizon_s=60.0)

        assert df_urgence["action_class_v2"].iloc[0] >= \
               df_sans_urgence["action_class_v2"].iloc[0]


# ---------------------------------------------------------------------------
# Tests oracle v2 : cas limites
# ---------------------------------------------------------------------------

class TestOracleV2CasLimites:

    def test_machine_froide_stable_classe_zero(self):
        """Machine à T=44°C (50% de T_shutdown=88) sans trajectoire -> classe 0."""
        # temp_ratio = clip((44 - 44) / 44, 0, 1) = 0
        df = make_df(temperature_c=44.0, temp_delta_30s=0.0)
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert df["action_class_v2"].iloc[0] == 0

    def test_machine_chaude_stable_classe_elevee(self):
        """Machine proche du shutdown sans trajectoire -> classe élevée."""
        df = make_df(temperature_c=85.0, temp_delta_30s=0.0)
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert df["action_class_v2"].iloc[0] >= N_LEVELS - 2

    def test_status_degraded_force_classe_max(self):
        """status=degraded doit toujours forcer la classe max."""
        df = make_df(temperature_c=50.0, temp_delta_30s=-2.0, status="degraded")
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert df["action_class_v2"].iloc[0] == N_LEVELS - 1

    def test_sans_features_trajectoire_degrade_gracieusement(self):
        """Sans temp_delta_* ni time_to_failure_s, oracle v2 se degrade vers v1."""
        df = pd.DataFrame({
            "temperature_c": [70.0],
            "status": ["on"],
        })
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert "action_class_v2" in df.columns
        assert 0 <= df["action_class_v2"].iloc[0] <= N_LEVELS - 1

    def test_valeurs_dans_rpm_levels(self):
        """Toutes les action_class_v2 doivent être dans [0, n_levels-1]."""
        temps = np.linspace(30, 87, 50)
        deltas = np.linspace(-5, 5, 50)
        df = pd.DataFrame({
            "temperature_c": temps,
            "temp_delta_30s": deltas,
            "status": ["on"] * 50,
            "time_to_failure_s": np.where(np.arange(50) % 10 == 0,
                                           np.linspace(5, 120, 50), np.nan),
        })
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert df["action_class_v2"].between(0, N_LEVELS - 1).all()

    def test_batch_grande_taille(self):
        """L'oracle v2 doit fonctionner sur un grand DataFrame sans erreur."""
        n = 10_000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "temperature_c": rng.uniform(30, 88, n),
            "temp_delta_30s": rng.uniform(-8, 8, n),
            "status": rng.choice(["on", "degraded"], n, p=[0.95, 0.05]),
            "time_to_failure_s": np.where(rng.random(n) < 0.3,
                                           rng.uniform(5, 120, n), np.nan),
        })
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert len(df) == n
        assert df["action_class_v2"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Tests de non-régression oracle v1
# ---------------------------------------------------------------------------

class TestNonRegressionOracleV1:

    def test_add_control_labels_v1_inchange(self):
        """add_control_labels (v1) doit produire les mêmes résultats qu'avant."""
        df = pd.DataFrame({
            "temperature_c": [44.0, 60.0, 75.0, 85.0],
            "status": ["on", "on", "on", "degraded"],
        })
        df = add_control_labels(df, t_shutdown_c=T_SHUTDOWN)
        assert list(df["action_class"]) == [0, 1, 3, 4]

    def test_v1_et_v2_coexistent_sans_conflit(self):
        """Les deux fonctions peuvent être appelées en séquence sans écraser les colonnes."""
        df = pd.DataFrame({
            "temperature_c": [60.0, 75.0],
            "temp_delta_30s": [3.0, -2.0],
            "status": ["on", "on"],
            "time_to_failure_s": [np.nan, np.nan],
        })
        df = add_control_labels(df, t_shutdown_c=T_SHUTDOWN)
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN)
        assert "action_class" in df.columns
        assert "action_class_v2" in df.columns

    def test_label_names_control_v2(self):
        """label_names_control_v2() doit retourner la liste correcte."""
        from features.labeler import label_names_control_v2
        assert "action_class_v2" in label_names_control_v2()

    def test_label_names_control_v1_inchange(self):
        """label_names_control() ne doit pas inclure action_class_v2."""
        assert "action_class_v2" not in label_names_control()


# ---------------------------------------------------------------------------
# Tests paramétrés : sensibilité aux paramètres
# ---------------------------------------------------------------------------

class TestParametres:

    @pytest.mark.parametrize("alpha,beta,gamma", [
        (1.0, 0.0, 0.0),   # purement temperature
        (0.0, 1.0, 0.0),   # purement vitesse
        (0.0, 0.0, 1.0),   # purement urgence
        (0.5, 0.3, 0.2),   # defaut
    ])
    def test_parametres_valides_pas_derreur(self, alpha, beta, gamma):
        """Tous les jeux de paramètres valides doivent fonctionner."""
        df = make_df(temperature_c=65.0, temp_delta_30s=2.0, time_to_failure_s=40.0)
        df = add_control_labels_v2(df, t_shutdown_c=T_SHUTDOWN,
                                   alpha=alpha, beta=beta, gamma=gamma)
        assert "action_class_v2" in df.columns
        assert 0 <= df["action_class_v2"].iloc[0] <= N_LEVELS - 1

    def test_alpha_zero_vitesse_seule_montee(self):
        """Avec alpha=0, gamma=0 : seule la vitesse thermique compte."""
        df_montee = make_df(temperature_c=50.0, temp_delta_30s=5.0)
        df_stable = make_df(temperature_c=50.0, temp_delta_30s=0.0)

        df_montee = add_control_labels_v2(df_montee, t_shutdown_c=T_SHUTDOWN,
                                          alpha=0.0, beta=1.0, gamma=0.0)
        df_stable = add_control_labels_v2(df_stable, t_shutdown_c=T_SHUTDOWN,
                                          alpha=0.0, beta=1.0, gamma=0.0)

        assert df_montee["action_class_v2"].iloc[0] > df_stable["action_class_v2"].iloc[0]

    def test_gamma_zero_urgence_ignoree(self):
        """Avec gamma=0, la composante urgence ne contribue pas au score.

        Note : la hard rule (ttf < horizon/3) reste active indépendamment de gamma.
        On utilise ttf=200s >> horizon=60s pour éviter de déclencher la hard rule.
        """
        df_sans = make_df(temperature_c=55.0, temp_delta_30s=0.0)
        # ttf=200s > horizon/3=20s -> hard rule inactive -> seul gamma contribue à score
        df_avec = make_df(temperature_c=55.0, temp_delta_30s=0.0, time_to_failure_s=200.0)

        df_sans = add_control_labels_v2(df_sans, t_shutdown_c=T_SHUTDOWN,
                                        alpha=0.5, beta=0.5, gamma=0.0, horizon_s=60.0)
        df_avec = add_control_labels_v2(df_avec, t_shutdown_c=T_SHUTDOWN,
                                        alpha=0.5, beta=0.5, gamma=0.0, horizon_s=60.0)

        # Sans gamma et sans hard rule, les deux doivent être identiques
        assert df_sans["action_class_v2"].iloc[0] == df_avec["action_class_v2"].iloc[0]
