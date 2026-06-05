---
name: dcm-is-rt-not-ut
description: DCM/DICOM format in this project is radiographic testing images, not ultrasonic
metadata:
  type: project
---

DCM/DICOM 格式图像在本项目中是**射线检测（RT, Radiographic Testing）**图像，而非超声检测（UT）。之前被错误理解为超声图像，已纠正。

**Why:** 用户明确指出 DCM 格式为射线检测图像类型。

**How to apply:** 在撰写代码注释、文档说明、增强算法描述时，提到图像类型应使用"射线/RT/radiographic"而非"超声/ultrasonic"。
