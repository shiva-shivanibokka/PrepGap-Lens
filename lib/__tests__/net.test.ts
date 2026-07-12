import { describe, it, expect } from "vitest";
import { isPrivateIp } from "../net";

describe("isPrivateIp (SSRF guard)", () => {
  it("flags loopback / private / link-local / metadata / unique-local", () => {
    for (const ip of [
      "127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "172.31.255.255",
      "169.254.169.254", "::1", "fe80::1", "fc00::1", "fd12::1", "0.0.0.0", "::ffff:127.0.0.1",
    ])
      expect(isPrivateIp(ip), ip).toBe(true);
  });

  it("allows genuinely public addresses", () => {
    for (const ip of ["8.8.8.8", "1.1.1.1", "93.184.216.34", "172.15.0.1", "172.32.0.1", "2606:4700::1111"])
      expect(isPrivateIp(ip), ip).toBe(false);
  });
});
