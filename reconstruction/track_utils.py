"""Shared trajectory reporting helpers."""


def intervals(indices, times):
    groups = []
    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return [dict(first_frame=int(g[0]), last_frame=int(g[-1]),
                 start_s=float(times[g[0]]), end_s=float(times[g[-1]])) for g in groups]
