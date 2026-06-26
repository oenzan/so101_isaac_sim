from pxr import Usd, UsdGeom, Gf
stage = Usd.Stage.CreateInMemory()
prim = stage.DefinePrim("/robot", "Xform")
xform = UsdGeom.Xformable(prim)
xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 180.0))
print(stage.GetRootLayer().ExportToString())
