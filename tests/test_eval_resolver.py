"""arXiv 编号核实的测试：不联网，替换掉 arXiv 客户端。"""

from __future__ import annotations

from datetime import datetime, timezone

import arxiv

from scholargraph.evaluation.resolver import ArxivIdResolver

EXISTING = {"2005.11401": "Retrieval-Augmented Generation", "2309.03883": "DoLa"}


def arxiv_result(number: str, title: str) -> arxiv.Result:
    published = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return arxiv.Result(
        entry_id=f"http://arxiv.org/abs/{number}v2",
        updated=published,
        published=published,
        title=title,
        authors=[arxiv.Result.Author("Ada Lovelace")],
        summary=f"Abstract of {title}.",
    )


class FakeArxivClient:
    """只认识 EXISTING 里的编号；`fail` 为真时模拟网络故障。"""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.requested: list[list[str]] = []

    def results(self, search: arxiv.Search):
        self.requested.append(list(search.id_list))
        if self.fail:
            raise arxiv.HTTPError("http://export.arxiv.org/api/query", 3, 503)
        return iter(arxiv_result(n, EXISTING[n]) for n in search.id_list if n in EXISTING)


def make_resolver(fail: bool = False) -> tuple[ArxivIdResolver, FakeArxivClient]:
    resolver = ArxivIdResolver(min_interval_seconds=0)
    client = FakeArxivClient(fail)
    resolver._client = client
    return resolver, client


def test_existing_papers_are_found_and_missing_ones_are_not():
    resolver, client = make_resolver()

    resolution = resolver.resolve(["arXiv:2005.11401", "arXiv:9999.99999"])

    assert set(resolution.found) == {"arXiv:2005.11401"}
    assert resolution.found["arXiv:2005.11401"].title == "Retrieval-Augmented Generation"
    assert resolution.unreachable == set()  # 9999.99999 是确认不存在，不是没查到
    assert client.requested == [["2005.11401", "9999.99999"]]  # 一次请求批量查询


def test_malformed_and_non_arxiv_ids_are_not_sent_to_arxiv():
    resolver, client = make_resolver()

    resolution = resolver.resolve(["arXiv:not-a-number", "OpenAlex:W123", "S2:42"])

    assert resolution.found == {} and resolution.unreachable == set()
    assert client.requested == []  # 格式不合法的编号会让整个请求报错，所以根本不发


def test_network_failure_is_reported_as_unreachable_not_as_missing():
    resolver, _ = make_resolver(fail=True)

    resolution = resolver.resolve(["arXiv:2005.11401"])

    assert resolution.found == {}
    assert resolution.unreachable == {"arXiv:2005.11401"}


def test_unreachable_ids_are_retried_next_time():
    resolver, client = make_resolver(fail=True)
    resolver.resolve(["arXiv:2005.11401"])

    client.fail = False
    resolution = resolver.resolve(["arXiv:2005.11401"])

    assert set(resolution.found) == {"arXiv:2005.11401"}


def test_each_id_is_looked_up_only_once():
    resolver, client = make_resolver()

    resolver.resolve(["arXiv:2005.11401", "arXiv:9999.99999"])
    resolution = resolver.resolve(["arXiv:2005.11401", "arXiv:9999.99999", "arXiv:2309.03883"])

    assert set(resolution.found) == {"arXiv:2005.11401", "arXiv:2309.03883"}
    assert client.requested == [["2005.11401", "9999.99999"], ["2309.03883"]]
